#!/usr/bin/env ruby
# frozen_string_literal: true

require "yaml"
require "date"
require "pathname"
require "bigdecimal"

module TrainingVaultValidator
  DEFAULT_ROOT = Pathname.new(__dir__).join("../..").cleanpath
  PHASES = %w[data structure content all].freeze
  PROBLEM_FIELDS = %w[题号 母题 母题模块 章节 来源 引用 标题 优先级 限时 手写 训练目标 状态 首刷日期 首刷耗时 首刷结果 二刷日期 二刷结果 纸笔日期 纸笔得分 错误类型 下次复习 复习次数 掌握评分 薄弱点 备注 证据状态 最后训练 思想得分 正确性得分 代码得分 复杂度得分].freeze
  PATTERN_FIELDS = %w[模板状态 上次复写 下次复写 复写得分].freeze
  STATUSES = %w[未开始 学习中 已AC 已二刷 可纸笔].freeze
  EVIDENCE_STATES = %w[有效 待校准].freeze
  SOURCES = %w[408真题 LeetCode].freeze
  DATE_FIELDS = %w[首刷日期 二刷日期 纸笔日期 下次复习 最后训练].freeze
  SCORE_FIELDS = %w[思想得分 正确性得分 代码得分 复杂度得分 纸笔得分].freeze
  Note = Struct.new(:path, :frontmatter, :body, :valid, keyword_init: true)

  class Runner
    def initialize(root:, out:, err:)
      @root = root.is_a?(Pathname) ? root : Pathname.new(root.to_s)
      @out, @err, @errors = out, err, []
    end

    def run(argv)
      phase = phase_from(argv)
      return 1 unless phase
      problem_paths = markdown_paths(@root.join("刷题记录"))
      pattern_paths = markdown_paths(@root.join("母题模板"))
      problems = load_notes(problem_paths)
      patterns = load_notes(pattern_paths)
      @errors << "problem count is #{problem_paths.length}; expected 57" unless problem_paths.length == 57
      @errors << "pattern count is #{pattern_paths.length}; expected 12" unless pattern_paths.length == 12
      problems.each { |note| validate_problem(note, phase) if note.valid }
      patterns.each { |note| validate_pattern(note, phase) if note.valid }
      exam_count = problems.count { |note| note.frontmatter["来源"] == "408真题" }
      pending_count = problems.count { |note| note.frontmatter["证据状态"] == "待校准" }
      @errors << "exam problem count is #{exam_count}; expected 10" unless exam_count == 10
      @errors << "pending evidence count is #{pending_count}; expected range 0..15" unless (0..15).cover?(pending_count)
      validate_pages if %w[structure content all].include?(phase)
      if @errors.empty?
        @out.puts "PASS phase=#{phase} problems=#{problems.length} exams=#{exam_count} patterns=#{patterns.length} pending=#{pending_count}"
        0
      else
        @errors.each { |error| @err.puts "ERROR #{error}" }
        @err.puts "FAIL phase=#{phase} errors=#{@errors.length}"
        1
      end
    rescue StandardError => e
      @err.puts "ERROR validator internal failure: #{e.class}: #{e.message}"
      @err.puts "FAIL phase=#{phase || "unknown"} errors=#{@errors.length + 1}"
      1
    end

    private

    def phase_from(argv)
      arg = argv.first
      phase = arg&.start_with?("--phase=") ? arg.split("=", 2).last : (arg || "all")
      return phase if PHASES.include?(phase)
      @err.puts "Invalid phase #{phase.inspect}; accepted phases: #{PHASES.join(", ")}"
      nil
    end

    def markdown_paths(directory)
      directory.directory? ? directory.glob("*.md").sort : []
    end

    def load_notes(paths)
      paths.map do |path|
        parsed = parse_note(path)
        parsed && Note.new(path: path, frontmatter: parsed[0], body: parsed[1], valid: true)
      end.compact
    end

    def parse_note(path)
      raw = begin
        path.binread
      rescue SystemCallError => e
        @errors << "#{path}: cannot read file (#{e.message})"
        return nil
      end
      raw.force_encoding(Encoding::UTF_8)
      unless raw.valid_encoding?
        @errors << "#{path}: file is not valid UTF-8"
        return nil
      end
      unless raw.start_with?("---\n", "---\r\n")
        @errors << "#{path}: missing leading YAML frontmatter"
        return nil
      end
      match = raw.match(/\A---\r?\n(.*?)\r?\n---\r?\n?(.*)\z/m)
      unless match
        @errors << "#{path}: YAML frontmatter is not closed with ---"
        return nil
      end
      begin
        fm = YAML.safe_load(match[1], permitted_classes: [Date], aliases: false)
        unless fm.is_a?(Hash)
          @errors << "#{path}: YAML frontmatter must be a mapping"
          return nil
        end
        [fm, match[2]]
      rescue Psych::Exception => e
        @errors << "#{path}: invalid YAML frontmatter (#{e.message.lines.first.strip})"
        nil
      rescue EncodingError => e
        @errors << "#{path}: invalid frontmatter encoding (#{e.message})"
        nil
      end
    end

    def scalar?(value)
      !value.is_a?(Array) && !value.is_a?(Hash)
    end

    def blank?(value)
      value.nil? || value == ""
    end

    def validate_scalar_fields(note, fields)
      fields.each { |field| @errors << "#{note.path}: #{field} must be scalar" if note.frontmatter.key?(field) && !scalar?(note.frontmatter[field]) }
    end

    def validate_problem(note, phase)
      fm = note.frontmatter
      PROBLEM_FIELDS.each { |field| @errors << "#{note.path}: missing problem field #{field}" unless fm.key?(field) }
      validate_scalar_fields(note, PROBLEM_FIELDS)
      @errors << "#{note.path}: 来源 must be exactly 408真题 or LeetCode" unless SOURCES.include?(fm["来源"])
      @errors << "#{note.path}: invalid 状态 #{fm["状态"].inspect}" unless STATUSES.include?(fm["状态"])
      @errors << "#{note.path}: invalid 证据状态 #{fm["证据状态"].inspect}" unless EVIDENCE_STATES.include?(fm["证据状态"])
      @errors << "#{note.path}: cssclasses must contain ds-408-problem" unless Array(fm["cssclasses"]).include?("ds-408-problem")
      validate_dates(note)
      validate_scores(note)
      return unless %w[structure content all].include?(phase)
      @errors << "#{note.path}: missing state control view" unless note.body.include?('dv.view("11_数据结构/系统/状态控件"')
      @errors << "#{note.path}: missing question navigation view" unless note.body.include?('dv.view("11_数据结构/系统/题目导航"')
      @errors << "#{note.path}: contains duplicated legacy status marker" if note.body.include?("const statuses = ['未开始'")
      @errors << "#{note.path}: contains static status conflict" if note.body.include?("当前状态：**未开始**")
      return unless %w[content all].include?(phase)
      %w[独立作答 训练控制台 错因复盘].each { |label| @errors << "#{note.path}: missing content label #{label}" unless note.body.include?(label) }
      compact = note.body.gsub(/\s+/, "")
      if fm["来源"] == "408真题"
        %w[算法思想 正确性说明 C/C++代码 复杂度分析 评分点].each { |label| @errors << "#{note.path}: missing 408 content label #{label}" unless compact.include?(label) }
        @errors << "#{note.path}: missing success callout [!success]-" unless note.body.include?("[!success]-")
      else
        @errors << "#{note.path}: missing LeetCode label 我的实现与复盘" unless note.body.include?("我的实现与复盘")
      end
    end

    def validate_dates(note)
      DATE_FIELDS.each do |field|
        value = note.frontmatter[field]
        next if blank?(value) || value.is_a?(Date)
        valid = value.is_a?(String) && begin
          Date.iso8601(value)
          true
        rescue ArgumentError
          false
        end
        @errors << "#{note.path}: #{field} must be empty, Date, or ISO YYYY-MM-DD" unless valid
      end
    end

    def decimal(value)
      return nil unless scalar?(value) && !blank?(value) && !value.is_a?(TrueClass) && !value.is_a?(FalseClass)
      BigDecimal(value.to_s)
    rescue ArgumentError
      nil
    end

    def validate_scores(note)
      fm = note.frontmatter
      present = SCORE_FIELDS.map { |field| !blank?(fm[field]) }
      return if present.none?
      unless present.all?
        @errors << "#{note.path}: all five score fields must be blank or all present"
        return
      end
      values = SCORE_FIELDS.map { |field| decimal(fm[field]) }
      ranges = [BigDecimal("2"), BigDecimal("2"), BigDecimal("4"), BigDecimal("2")]
      values[0, 4].each_with_index do |value, index|
        @errors << "#{note.path}: #{SCORE_FIELDS[index]} must be numeric" unless value
        @errors << "#{note.path}: #{SCORE_FIELDS[index]} out of range" if value && (value < 0 || value > ranges[index])
      end
      @errors << "#{note.path}: 纸笔得分 must be numeric" unless values[4]
      @errors << "#{note.path}: 纸笔得分 must equal component sum" if values.all? && values[4] != values[0, 4].sum
    end

    def validate_pattern(note, phase)
      PATTERN_FIELDS.each { |field| @errors << "#{note.path}: missing pattern field #{field}" unless note.frontmatter.key?(field) }
      validate_scalar_fields(note, PATTERN_FIELDS)
      @errors << "#{note.path}: cssclasses must contain ds-408-pattern" unless Array(note.frontmatter["cssclasses"]).include?("ds-408-pattern")
      return unless %w[content all].include?(phase)
      %w[识别信号 核心不变量 C/C++纸笔模板 正确性说明 复杂度 常见边界与失分点 可迁移变化 模板闭卷复写].each { |label| @errors << "#{note.path}: missing pattern content label #{label}" unless note.body.include?(label) }
      @errors << "#{note.path}: contains legacy handwriting placeholder" if note.body.include?("在此手写核心模板代码")
    end

    def safe_text(path)
      raw = path.binread.force_encoding(Encoding::UTF_8)
      raise EncodingError, "invalid UTF-8" unless raw.valid_encoding?
      raw
    rescue SystemCallError, EncodingError => e
      @errors << "#{path}: cannot read text (#{e.message})"
      nil
    end

    def validate_pages
      %w[408作战台.md 总览.md 错题回炉.md 训练数据校准.md 真题/真题冲刺.md].each { |relative| @errors << "missing required page #{@root.join(relative)}" unless @root.join(relative).file? }
      overview = @root.join("总览.md")
      if (text = safe_text(overview))
        %w[系统/统计卡 系统/复习队列 系统/题库表].each { |label| @errors << "#{overview}: missing component view path #{label}" unless text.include?(label) }
      end
      dashboard = @root.join("408作战台.md")
      if safe_text(dashboard)
        parsed = parse_note(dashboard)
        fm = parsed && parsed[0]
        @errors << "#{dashboard}: missing dashboard: true" unless fm && fm["dashboard"] == true
        @errors << "#{dashboard}: missing type: dataview" unless fm && fm["type"] == "dataview"
      end
    end
  end

  def self.main(argv: ARGV, root: DEFAULT_ROOT, out: $stdout, err: $stderr)
    Runner.new(root: root, out: out, err: err).run(argv)
  end
end

exit(TrainingVaultValidator.main) if $PROGRAM_NAME == __FILE__
