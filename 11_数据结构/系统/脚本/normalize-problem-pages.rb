#!/usr/bin/env ruby
# frozen_string_literal: true

require "yaml"
require "psych"
require "date"
require "pathname"
require "optparse"
require "securerandom"

# Rebuilds only the body of the 57 problem notes.  Frontmatter is validated but
# never serialized, so its original bytes remain untouched.
module NormalizeProblemPages
  DEFAULT_ROOT = Pathname.new(__dir__).join("../..").cleanpath
  LEGACY_HEADINGS = ["## ⚡ 状态快速切换", "## 🎯 训练目标", "## 📊 状态追踪", "## 📝 解题笔记", "## 📚 错题复盘"].freeze
  PROBLEM_VIEW = 'await dv.view("11_数据结构/系统/状态控件", { mode: "problem" });'
  NAVIGATION_VIEW = 'await dv.view("11_数据结构/系统/题目导航");'
  Note = Struct.new(:path, :frontmatter_bytes, :body, :source, :line_ending, :mode, keyword_init: true)
  Write = Struct.new(:path, :contents, :mode, keyword_init: true)
  Record = Struct.new(:write, :stage, :backup, :backed_up, :installed, keyword_init: true)

  class NormalizeError < StandardError; end

  class FileOperations
    def write_stage(path, contents, mode)
      File.open(path, "wb") do |file|
        file.write(contents)
        file.chmod(mode)
        file.flush
        file.fsync
      end
    end

    def rename(from, to)
      File.rename(from, to)
    end

    def delete(path)
      File.delete(path) if File.exist?(path)
    end

    def mode(path)
      File.stat(path).mode & 0o777
    end

    def sync_directory(path)
      File.open(File.dirname(path), "r") { |directory| directory.fsync }
    rescue Errno::EACCES, Errno::EINVAL, Errno::EPERM, NotImplementedError
      nil
    end
  end

  class Runner
    def initialize(root:, fixture:, out:, err:, file_operations: FileOperations.new)
      @root, @fixture, @out, @err, @file_operations = Pathname.new(root.to_s), fixture, out, err, file_operations
    end

    def run
      notes = load_notes
      validate_counts!(notes)
      writes = notes.map { |note| normalized_write(note) }.compact
      transactional_write(writes)
      leetcode = notes.count { |note| note.source == "LeetCode" }
      exams = notes.count { |note| note.source == "408真题" }
      @out.puts "normalized=#{writes.length} scanned=#{notes.length} leetcode=#{leetcode} exams=#{exams}"
      0
    rescue NormalizeError, SystemCallError, IOError => e
      @err.puts "ERROR #{e.message}"
      1
    end

    private

    def load_notes
      directory = @root.join("刷题记录")
      raise NormalizeError, "#{directory}: directory does not exist" unless directory.directory?

      directory.glob("*.md").sort.map { |path| load_note(path) }
    end

    def validate_counts!(notes)
      return if @fixture
      raise NormalizeError, "problem count is #{notes.length}; expected 57" unless notes.length == 57
      counts = notes.each_with_object(Hash.new(0)) { |note, result| result[note.source] += 1 }
      raise NormalizeError, "LeetCode count is #{counts["LeetCode"] || 0}; expected 47" unless counts["LeetCode"] == 47
      raise NormalizeError, "408真题 count is #{counts["408真题"] || 0}; expected 10" unless counts["408真题"] == 10
    end

    def load_note(path)
      raw = path.binread
      raw.force_encoding(Encoding::UTF_8)
      raise NormalizeError, "#{path}: file is not valid UTF-8" unless raw.valid_encoding?
      match = raw.match(/\A(---(\r?\n)(.*?)\r?\n---\r?\n)(.*)\z/m)
      raise NormalizeError, "#{path}: missing or malformed leading YAML frontmatter" unless match

      yaml = match[3]
      reject_duplicate_yaml_keys!(yaml, path)
      frontmatter = YAML.safe_load(yaml, permitted_classes: [Date], aliases: false)
      raise NormalizeError, "#{path}: YAML frontmatter must be a mapping" unless frontmatter.is_a?(Hash)
      source = frontmatter["来源"]
      unless ["LeetCode", "408真题"].include?(source)
        raise NormalizeError, "#{path}: 来源 must be LeetCode or 408真题"
      end
      Note.new(path: path, frontmatter_bytes: match[1], body: match[4], source: source,
               line_ending: match[2], mode: @file_operations.mode(path))
    rescue Psych::Exception => e
      raise NormalizeError, "#{path}: invalid YAML frontmatter (#{e.message.lines.first.strip})"
    end

    def reject_duplicate_yaml_keys!(yaml, path)
      root = Psych.parse(yaml)
      walk_yaml!(root, path)
    rescue Psych::SyntaxError => e
      raise NormalizeError, "#{path}: invalid YAML frontmatter (#{e.message.lines.first.strip})"
    end

    def walk_yaml!(node, path)
      case node
      when Psych::Nodes::Document, Psych::Nodes::Sequence
        node.children.each { |child| walk_yaml!(child, path) }
      when Psych::Nodes::Mapping
        seen = {}
        node.children.each_slice(2) do |key, value|
          raise NormalizeError, "#{path}: malformed YAML mapping" unless key && value && key.is_a?(Psych::Nodes::Scalar)
          name = key.value
          raise NormalizeError, "#{path}: duplicate YAML key #{name.inspect}" if seen[name]
          seen[name] = true
          walk_yaml!(value, path)
        end
      end
    end

    def normalized_write(note)
      if standardized?(note)
        return repair_legacy_prefix_spacing(note)
      end
      prefix, goal, notes = parse_legacy(note)
      Write.new(path: note.path, contents: note.frontmatter_bytes + build_body(prefix, goal, notes, note.source, note.line_ending), mode: note.mode)
    end

    def parse_legacy(note)
      boundaries = LEGACY_HEADINGS.map do |heading|
        positions = top_level_positions(note.body, heading)
        raise NormalizeError, "#{note.path}: missing legacy boundary #{heading}" if positions.empty?
        raise NormalizeError, "#{note.path}: duplicate legacy boundary #{heading}" if positions.length > 1
        positions.first
      end
      unless boundaries.each_cons(2).all? { |left, right| left < right }
        raise NormalizeError, "#{note.path}: legacy boundaries are out of order"
      end
      prefix = utf8_slice(note.body, 0, boundaries[0])
      goal = content_between(note.body, boundaries[1], boundaries[2])
      notes = content_between(note.body, boundaries[3], boundaries[4])
      review_start = boundary_line_end(note.body, boundaries[4])
      review = utf8_slice(note.body, review_start, note.body.length - review_start)
      validate_legacy_review!(review, note.path)
      [prefix, goal, notes]
    end

    def top_level_positions(body, heading)
      body.to_enum(:scan, /^#{Regexp.escape(heading)}\r?$/).map { Regexp.last_match.begin(0) }
    end

    def boundary_line_end(body, position)
      newline = body.index("\n", position)
      raise NormalizeError, "legacy boundary is not terminated" unless newline
      newline + 1
    end

    def content_between(body, heading_position, next_heading_position)
      start = boundary_line_end(body, heading_position)
      trim_separator_blank_lines(utf8_slice(body, start, next_heading_position - start))
    end

    def trim_separator_blank_lines(text)
      text.sub(/\A(?:[ \t]*\r?\n)+/, "").sub(/(?:\r?\n[ \t]*)+\z/, "")
    end

    def utf8_slice(text, start, length)
      text[start, length]
    end

    def validate_legacy_review!(review, path)
      lines = review.split(/\r?\n/, -1).reject { |line| line.match?(/\A[ \t]*\z/) }
      patterns = [
        /\A- \*\*错误类型\*\*：.*\z/,
        /\A- \*\*薄弱点\*\*：.*\z/,
        /\A- \*\*复习次数\*\*：.*，\*\*下次复习\*\*：.*\z/,
        /\A- \*\*备注\*\*：.*\z/
      ]
      unless lines.length == patterns.length && lines.zip(patterns).all? { |line, pattern| pattern.match?(line) }
        raise NormalizeError, "#{path}: unclassified content after ## 📚 错题复盘"
      end
    end

    def build_body(prefix, goal, notes, source, eol)
      parts = ["## 🎯 训练目标#{eol}#{goal}"]
      parts << independent_answer(eol)
      parts << "## 🧭 训练控制台#{eol}#{fence(PROBLEM_VIEW, eol)}"
      parts << (source == "LeetCode" ? "## 📝 我的实现与复盘#{eol}#{notes}" : exam_answer(notes, eol))
      parts << review(eol)
      parts << "## 🧭 导航#{eol}#{fence(NAVIGATION_VIEW, eol)}"
      prefix + parts.reject(&:empty?).join(eol + eol) + eol
    end

    def independent_answer(eol)
      ["## ✍️ 独立作答", "> [!question] 闭卷作答区", "> 先写输入、输出、约束和朴素方法，再写关键不变量、边界与复杂度。完成后再展开参考内容。", "",
       "- 算法思想：", "- 关键不变量或递归含义：", "- 边界条件：", "- 时间复杂度及依据：", "- 空间复杂度及依据："].join(eol)
    end

    def fence(code, eol)
      ["```dataviewjs", code, "```"].join(eol)
    end

    def exam_answer(notes, eol)
      quoted = notes.split(/\r?\n/, -1).map { |line| line.empty? ? ">" : "> #{line}" }.join(eol)
      "## 🧠 满分参考答案#{eol}> [!success]- 展开满分参考答案（完成闭卷作答后再看）#{eol}#{quoted}"
    end

    def review(eol)
      ["## 🔁 错因复盘", "- 错误类型：以顶部属性为准。", "- 薄弱点：以顶部属性为准。", "- 下一步：根据控制台下次复习日期闭卷重做。"].join(eol)
    end

    def standardized?(note)
      return false if (LEGACY_HEADINGS - ["## 🎯 训练目标"]).any? { |heading| note.body.include?(heading) }
      source_heading = note.source == "LeetCode" ? "## 📝 我的实现与复盘" : "## 🧠 满分参考答案"
      expected = ["## 🎯 训练目标", "## ✍️ 独立作答", "## 🧭 训练控制台", source_heading, "## 🔁 错因复盘", "## 🧭 导航"]
      headings = note.body.to_enum(:scan, /^## .*?\r?$/).map { Regexp.last_match[0].delete_suffix("\r") }
      first = headings.index("## 🎯 训练目标")
      return false unless first && headings[first, headings.length - first] == expected
      return false unless note.body.scan('dv.view("11_数据结构/系统/状态控件"').length == 1
      return false unless note.body.scan('dv.view("11_数据结构/系统/题目导航"').length == 1
      return false unless note.body.include?(PROBLEM_VIEW) && note.body.include?(NAVIGATION_VIEW)
      return false unless note.body.include?("> [!question] 闭卷作答区") && note.body.include?("- 算法思想：")
      return false unless note.body.include?("## 🔁 错因复盘") && note.body.include?("- 错误类型：以顶部属性为准。")
      return false if note.body.include?("const statuses = ['未开始'") || note.body.include?("当前状态：**未开始**")
      note.source != "408真题" || note.body.include?("> [!success]- 展开满分参考答案（完成闭卷作答后再看）")
    end

    # Earlier local drafts collapsed the original extra separator line before
    # the training heading.  LeetCode legacy cards consistently use three
    # terminal newlines; repair that historical draft once while retaining
    # idempotence for every subsequently normalized page.
    def repair_legacy_prefix_spacing(note)
      return nil unless note.source == "LeetCode"
      goal_position = note.body.index("## 🎯 训练目标")
      prefix = note.body[0...goal_position]
      eol = note.line_ending
      return nil unless prefix.end_with?(eol + eol) && !prefix.end_with?(eol + eol + eol)

      Write.new(path: note.path, contents: note.frontmatter_bytes + prefix + eol + note.body[goal_position..-1], mode: note.mode)
    end

    def transactional_write(writes)
      return if writes.empty?
      records = writes.map do |write|
        stage = "#{write.path}.normalize-stage-#{Process.pid}-#{SecureRandom.hex(6)}"
        @file_operations.write_stage(stage, write.contents, write.mode)
        Record.new(write: write, stage: stage, backup: "#{write.path}.normalize-backup-#{Process.pid}-#{SecureRandom.hex(6)}", backed_up: false, installed: false)
      end
      records.each do |record|
        @file_operations.rename(record.write.path, record.backup)
        record.backed_up = true
        @file_operations.rename(record.stage, record.write.path)
        record.installed = true
        @file_operations.sync_directory(record.write.path)
      end
      records.each { |record| @file_operations.delete(record.backup) }
    rescue SystemCallError, IOError => error
      records&.reverse_each do |record|
        begin
          @file_operations.delete(record.write.path) if record.installed
          @file_operations.rename(record.backup, record.write.path) if record.backed_up && File.exist?(record.backup)
        rescue SystemCallError, IOError
          # Preserve the original triggering error; all remaining records are still attempted.
        end
      end
      raise error
    ensure
      records&.each do |record|
        @file_operations.delete(record.stage) rescue nil
        @file_operations.delete(record.backup) rescue nil
      end
    end
  end

  def self.main(argv: ARGV, root: DEFAULT_ROOT, out: $stdout, err: $stderr, file_operations: FileOperations.new)
    fixture = false
    parser = OptionParser.new do |options|
      options.on("--root ROOT") { |value| root = value }
      options.on("--fixture") { fixture = true }
    end
    parser.parse!(argv)
    Runner.new(root: root, fixture: fixture, out: out, err: err, file_operations: file_operations).run
  rescue OptionParser::ParseError => e
    err.puts "ERROR #{e.message}"
    1
  end
end

exit(NormalizeProblemPages.main) if $PROGRAM_NAME == __FILE__
