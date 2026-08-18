#!/usr/bin/env ruby
# frozen_string_literal: true

require "yaml"
require "date"
require "pathname"
require "optparse"

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
  Note = Struct.new(:path, :kind, :frontmatter, :body, :line_ending, keyword_init: true)

  class MigrationError < StandardError; end

  class Runner
    def initialize(root:, out:, err:)
      @root = Pathname.new(root.to_s)
      @out = out
      @err = err
    end

    def run
      notes = load_all_notes
      changed = notes.count { |note| migrate(note) }
      notes.select { |note| note.instance_variable_get(:@changed) }.each { |note| write_note(note) }
      problems = notes.count { |note| note.kind == :problem }
      patterns = notes.count { |note| note.kind == :pattern }
      @out.puts "changed=#{changed} scanned=#{notes.length} problems=#{problems} patterns=#{patterns}"
      0
    rescue MigrationError, SystemCallError => e
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
      note.instance_variable_set(:@changed, changed)
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

    def write_note(note)
      yaml = YAML.dump(note.frontmatter).sub(/\A---\s*\n/, "")
      note.path.binwrite("---#{note.line_ending}#{yaml}---#{note.line_ending}#{note.body}")
    rescue SystemCallError => e
      raise MigrationError, "#{note.path}: cannot write file (#{e.message})"
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
