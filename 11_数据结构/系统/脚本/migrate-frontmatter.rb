#!/usr/bin/env ruby
# frozen_string_literal: true

require "yaml"
require "psych"
require "date"
require "pathname"
require "optparse"
require "securerandom"

module TrainingFrontmatterMigration
  PROBLEM_DEFAULTS = {
    "最后训练" => "",
    "思想得分" => "",
    "正确性得分" => "",
    "代码得分" => "",
    "复杂度得分" => ""
  }.freeze
  PATTERN_DEFAULTS = {
    "模板状态" => "未开始",
    "上次复写" => "",
    "下次复写" => "",
    "复写得分" => ""
  }.freeze
  CLEANUP_ATTEMPTS = 3
  Note = Struct.new(:path, :kind, :frontmatter, :body, :line_ending, keyword_init: true)
  PreparedWrite = Struct.new(:path, :contents, keyword_init: true)
  TransactionRecord = Struct.new(:path, :contents, :stage, :backup, :mode, :backed_up, :installed, keyword_init: true)

  class MigrationError < StandardError; end

  class FileOperations
    def write_stage(path, contents, mode)
      File.open(path, "wb") do |file|
        file.write(contents)
        file.chmod(mode)
        file.flush
        file.fsync
      end
    end

    def hard_link(source, destination)
      File.link(source, destination)
    end

    def copy_backup(source, destination, mode = File.stat(source).mode & 0o777)
      File.open(source, "rb") do |input|
        File.open(destination, "wb") do |output|
          IO.copy_stream(input, output)
          output.chmod(mode)
          output.flush
          output.fsync
        end
      end
    end

    def mode(path)
      File.stat(path).mode & 0o777
    end

    def sync_directory(path)
      File.open(File.dirname(path.to_s), "r") { |directory| directory.fsync }
    rescue Errno::EACCES, Errno::EINVAL, Errno::EISDIR, Errno::EPERM, NotImplementedError
      # Some filesystems and platforms do not expose directory fsync through Ruby.
      nil
    end

    def rename(source, destination)
      File.rename(source, destination)
    end

    def delete(path)
      File.delete(path) if File.exist?(path)
    end

    def exist?(path)
      File.exist?(path)
    end
  end

  class Runner
    def initialize(root:, out:, err:, file_operations: FileOperations.new)
      @root = Pathname.new(root.to_s)
      @out = out
      @err = err
      @file_operations = file_operations
    end

    def run
      notes = load_all_notes
      changed_notes = notes.select { |note| migrate(note) }
      prepared_writes = changed_notes.map { |note| PreparedWrite.new(path: note.path, contents: serialized_note(note)) }
      transactional_write(prepared_writes)
      problems = notes.count { |note| note.kind == :problem }
      patterns = notes.count { |note| note.kind == :pattern }
      @out.puts "changed=#{changed_notes.length} scanned=#{notes.length} problems=#{problems} patterns=#{patterns}"
      0
    rescue MigrationError, SystemCallError, IOError => e
      @err.puts "ERROR #{e.message}"
      1
    end

    private

    def load_all_notes
      problem_paths = markdown_paths(@root.join("刷题记录"))
      pattern_paths = markdown_paths(@root.join("母题模板"))
      problem_paths.map { |path| load_note(path, :problem) } +
        pattern_paths.map { |path| load_note(path, :pattern) }
    end

    def markdown_paths(directory)
      raise MigrationError, "#{directory}: directory does not exist" unless directory.directory?

      directory.glob("*.md").sort
    end

    def load_note(path, kind)
      raw = path.binread
      raw.force_encoding(Encoding::UTF_8)
      raise MigrationError, "#{path}: file is not valid UTF-8" unless raw.valid_encoding?

      match = raw.match(/\A---(\r?\n)(.*?)\r?\n---(\r?\n)(.*)\z/m)
      raise MigrationError, "#{path}: missing or malformed leading YAML frontmatter" unless match

      reject_duplicate_yaml_keys!(match[2], path)
      frontmatter = begin
        YAML.safe_load(match[2], permitted_classes: [Date], aliases: false)
      rescue Psych::Exception => e
        raise MigrationError, "#{path}: invalid YAML frontmatter (#{e.message})"
      end
      unless frontmatter.is_a?(Hash)
        raise MigrationError, "#{path}: YAML frontmatter must be a mapping"
      end
      unless frontmatter.keys.all? { |key| key.is_a?(String) }
        raise MigrationError, "#{path}: YAML frontmatter keys must be strings"
      end

      Note.new(path: path, kind: kind, frontmatter: frontmatter, body: match[4], line_ending: match[1])
    rescue Errno::ENOENT, Errno::EACCES => e
      raise MigrationError, "#{path}: cannot read file (#{e.message})"
    end

    def reject_duplicate_yaml_keys!(yaml, path)
      document = Psych.parse(yaml)
      walk_yaml_node_for_duplicate_keys(document, path)
    rescue Psych::SyntaxError => e
      raise MigrationError, "#{path}: invalid YAML frontmatter (#{e.message})"
    end

    def walk_yaml_node_for_duplicate_keys(node, path)
      case node
      when Psych::Nodes::Document, Psych::Nodes::Sequence
        node.children.each { |child| walk_yaml_node_for_duplicate_keys(child, path) }
      when Psych::Nodes::Mapping
        seen = {}
        node.children.each_slice(2) do |key, value|
          raise MigrationError, "#{path}: malformed YAML mapping" unless key && value

          key_name = yaml_mapping_key_name(key, path)
          raise MigrationError, "#{path}: duplicate YAML key #{key_name.inspect}" if seen.key?(key_name)

          seen[key_name] = true
          walk_yaml_node_for_duplicate_keys(key, path)
          walk_yaml_node_for_duplicate_keys(value, path)
        end
      end
    end

    def yaml_mapping_key_name(key, path)
      unless key.is_a?(Psych::Nodes::Scalar)
        raise MigrationError, "#{path}: complex YAML mapping keys are not supported"
      end

      if key.tag && !yaml_string_tag?(key.tag)
        raise MigrationError, "#{path}: YAML mapping key #{key.value.inspect} must resolve to a string"
      end
      return key.value if key.quoted || yaml_string_tag?(key.tag)

      value = YAML.safe_load(key.value, permitted_classes: [Date], aliases: false)
      return value if value.is_a?(String)

      raise MigrationError, "#{path}: YAML mapping key #{key.value.inspect} must resolve to a string"
    rescue Psych::Exception => e
      raise MigrationError, "#{path}: invalid YAML mapping key #{key.value.inspect} (#{e.message})"
    end

    def yaml_string_tag?(tag)
      tag == "!!str" || tag == "tag:yaml.org,2002:str"
    end

    def migrate(note)
      changed = normalize_cssclasses(note.frontmatter, note.kind == :problem ? "ds-408-problem" : "ds-408-pattern", note.path)
      defaults = note.kind == :problem ? PROBLEM_DEFAULTS : PATTERN_DEFAULTS
      defaults.each do |key, value|
        next if note.frontmatter.key?(key)

        note.frontmatter[key] = value
        changed = true
      end
      if note.kind == :problem && !note.frontmatter.key?("证据状态")
        note.frontmatter["证据状态"] = evidence_state(note.frontmatter)
        changed = true
      end
      changed
    end

    def normalize_cssclasses(frontmatter, project_class, path)
      had_key = frontmatter.key?("cssclasses")
      changed = !had_key
      classes = if !had_key
                  frontmatter["cssclasses"] = []
                elsif frontmatter["cssclasses"].is_a?(String)
                  frontmatter["cssclasses"] = [frontmatter["cssclasses"]]
                  changed = true
                  frontmatter["cssclasses"]
                elsif frontmatter["cssclasses"].is_a?(Array)
                  frontmatter["cssclasses"]
                else
                  raise MigrationError, "#{path}: cssclasses must be a string or array"
                end
      unless classes.include?(project_class)
        classes << project_class
        changed = true
      end
      changed
    end

    def evidence_state(frontmatter)
      status = frontmatter["状态"]
      return "有效" if %w[未开始 学习中].include?(status)
      return "有效" if status == "已AC" && nonempty?(frontmatter["首刷日期"]) && %w[独立通过 订正通过].include?(frontmatter["首刷结果"])
      return "有效" if status == "已二刷" && nonempty?(frontmatter["二刷日期"]) && frontmatter["二刷结果"] == "独立通过"
      return "有效" if status == "可纸笔" && nonempty?(frontmatter["纸笔日期"]) && numeric_at_least_eight?(frontmatter["纸笔得分"])

      "待校准"
    end

    def nonempty?(value)
      value.respond_to?(:strip) ? !value.strip.empty? : !value.nil?
    end

    def numeric_at_least_eight?(value)
      number = Float(value)
      number.finite? && number >= 8
    rescue ArgumentError, TypeError
      false
    end

    def serialized_note(note)
      yaml = YAML.dump(note.frontmatter).sub(/\A---\s*\n/, "")
      "---#{note.line_ending}#{yaml}---#{note.line_ending}#{note.body}"
    rescue Psych::Exception => e
      raise MigrationError, "#{note.path}: cannot serialize YAML frontmatter (#{e.message})"
    end

    def transactional_write(prepared_writes)
      return if prepared_writes.empty?

      committed = false
      records = prepared_writes.map do |prepared|
        TransactionRecord.new(
          path: prepared.path,
          contents: prepared.contents,
          stage: artifact_path(prepared.path, ".tmp"),
          backup: artifact_path(prepared.path, ".backup"),
          mode: nil,
          backed_up: false,
          installed: false
        )
      end
      stage_records(records)
      replace_records(records)
      committed = true
      report_post_commit_cleanup(records)
    rescue SystemCallError, IOError => e
      if committed
        @err.puts "WARNING migration committed; post-commit cleanup interrupted: #{e.message}"
        return
      end

      rollback_errors = rollback_records(records || [])
      cleanup_errors = rollback_errors.empty? ? cleanup_transaction_artifacts(records || []) : cleanup_stages(records || [])
      detail = ([e.message] + rollback_errors + cleanup_errors).join("; ")
      raise MigrationError, "transaction failed: #{detail}"
    ensure
      cleanup_stages(records || []) if defined?(records)
    end

    def artifact_path(path, suffix)
      loop do
        candidate = Pathname.new("#{path}.migrate-frontmatter-#{SecureRandom.hex(12)}#{suffix}")
        return candidate unless @file_operations.exist?(candidate)
      end
    end

    def stage_records(records)
      records.each do |record|
        record.mode = @file_operations.mode(record.path)
        @file_operations.write_stage(record.stage, record.contents, record.mode)
      end
    end

    def replace_records(records)
      records.each do |record|
        create_backup(record)
        @file_operations.rename(record.stage, record.path)
        record.installed = true
        @file_operations.sync_directory(record.path)
      end
    end

    def create_backup(record)
      begin
        @file_operations.hard_link(record.path, record.backup)
      rescue SystemCallError, NotImplementedError
        @file_operations.copy_backup(record.path, record.backup, record.mode)
      end
      record.backed_up = true
      @file_operations.sync_directory(record.path)
    end

    def rollback_records(records)
      errors = []
      records.reverse_each do |record|
        begin
          if record.installed && @file_operations.exist?(record.backup)
            @file_operations.rename(record.backup, record.path)
            @file_operations.sync_directory(record.path)
          end
        rescue SystemCallError, IOError => e
          errors << "rollback #{record.path}: #{e.message}"
        end
      end
      errors
    end

    def cleanup_stages(records)
      cleanup_artifacts(records, :stage)
    end

    def cleanup_transaction_artifacts(records)
      cleanup_artifacts(records, :stage) + cleanup_artifacts(records, :backup)
    end

    def report_post_commit_cleanup(records)
      failures = cleanup_artifacts_with_retries(records, :backup)
      return if failures.empty?

      @err.puts "WARNING migration committed; backup cleanup incomplete: #{failures.join("; ")}"
    end

    def cleanup_artifacts_with_retries(records, attribute)
      pending = records.map { |record| record.public_send(attribute) }
      failures = []
      CLEANUP_ATTEMPTS.times do
        failures = cleanup_paths(pending)
        pending = pending.select { |path| @file_operations.exist?(path) }
        return [] if pending.empty?
      end
      failures.empty? ? pending.map { |path| "cleanup #{path}: artifact remains" } : failures
    rescue SystemCallError, IOError => e
      ["cleanup #{attribute}: #{e.message}"]
    end

    def cleanup_paths(paths)
      failures = []
      paths.each do |path|
        @file_operations.delete(path) if @file_operations.exist?(path)
      rescue SystemCallError, IOError => e
        failures << "cleanup #{path}: #{e.message}"
      end
      failures
    end

    def cleanup_artifacts(records, attribute)
      cleanup_paths(records.map { |record| record.public_send(attribute) })
    end
  end

  module_function

  def root_from(argv)
    root = Pathname.new(__dir__).join("../..").cleanpath
    OptionParser.new do |opts|
      opts.banner = "Usage: #{File.basename($PROGRAM_NAME)} [--root VAULT_ROOT]"
      opts.on("--root ROOT", "Vault root containing 刷题记录 and 母题模板") { |value| root = Pathname.new(value) }
    end.parse!(argv)
    raise MigrationError, "unexpected arguments: #{argv.join(" ")}" unless argv.empty?

    root
  rescue OptionParser::ParseError => e
    raise MigrationError, e.message
  end
end

if $PROGRAM_NAME == __FILE__
  root = TrainingFrontmatterMigration.root_from(ARGV)
  exit TrainingFrontmatterMigration::Runner.new(root: root, out: $stdout, err: $stderr).run
end
