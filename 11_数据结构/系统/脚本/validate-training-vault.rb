#!/usr/bin/env ruby
# frozen_string_literal: true

require "yaml"
require "date"
require "pathname"

PROJECT_ROOT = Pathname.new(__dir__).join("../..").cleanpath
PHASES = %w[data structure content all].freeze
PROBLEM_FIELDS = %w[
  题号 母题 母题模块 章节 来源 引用 标题 优先级 限时 手写 训练目标 状态 首刷日期 首刷耗时
  首刷结果 二刷日期 二刷结果 纸笔日期 纸笔得分 错误类型 下次复习 复习次数 掌握评分 薄弱点 备注
  证据状态 最后训练 思想得分 正确性得分 代码得分 复杂度得分
].freeze
PATTERN_FIELDS = %w[模板状态 上次复写 下次复写 复写得分].freeze
STATUSES = %w[未开始 学习中 已AC 已二刷 可纸笔].freeze
EVIDENCE_STATES = %w[有效 待校准].freeze

def selected_phase
  argument = ARGV.first
  phase = if argument&.start_with?("--phase=")
            argument.split("=", 2).last
          else
            argument || "all"
          end
  abort "Invalid phase #{phase.inspect}; accepted phases: #{PHASES.join(", ")}" unless PHASES.include?(phase)
  phase
end

def parse_note(path, errors)
  text = path.read
  unless text.start_with?("---\n") || text.start_with?("---\r\n")
    errors << "#{path}: missing leading YAML frontmatter"
    return [{}, ""]
  end
  match = text.match(/\A---\r?\n(.*?)\r?\n---\r?\n?(.*)\z/m)
  unless match
    errors << "#{path}: YAML frontmatter is not closed with ---"
    return [{}, ""]
  end
  begin
    frontmatter = YAML.safe_load(match[1], permitted_classes: [Date], aliases: false) || {}
    unless frontmatter.is_a?(Hash)
      errors << "#{path}: YAML frontmatter must be a mapping"
      frontmatter = {}
    end
    [frontmatter, match[2]]
  rescue Psych::Exception => e
    errors << "#{path}: invalid YAML frontmatter (#{e.message.lines.first&.strip})"
    [{}, match[2]]
  end
end

def empty_value?(value)
  value.nil? || (value.respond_to?(:empty?) && value.empty?)
end

def number(value)
  return nil if empty_value?(value)
  Float(value)
rescue ArgumentError, TypeError
  nil
end

def validate_problem(path, frontmatter, body, phase, errors)
  PROBLEM_FIELDS.each { |field| errors << "#{path}: missing problem field #{field}" unless frontmatter.key?(field) }
  status = frontmatter["状态"]
  errors << "#{path}: invalid 状态 #{status.inspect}; expected one of #{STATUSES.join(", ")}" unless STATUSES.include?(status)
  evidence = frontmatter["证据状态"]
  errors << "#{path}: invalid 证据状态 #{evidence.inspect}; expected one of #{EVIDENCE_STATES.join(", ")}" unless EVIDENCE_STATES.include?(evidence)
  css = Array(frontmatter["cssclasses"])
  errors << "#{path}: cssclasses must contain ds-408-problem" unless css.include?("ds-408-problem")

  components = %w[思想得分 正确性得分 代码得分 复杂度得分]
  values = components.map { |field| number(frontmatter[field]) }
  raw_present = components.any? { |field| !empty_value?(frontmatter[field]) }
  if raw_present
    components.each_with_index do |field, index|
      raw = frontmatter[field]
      errors << "#{path}: #{field} must be numeric and non-empty when score components are present" if values[index].nil?
    end
    ranges = [0..2, 0..2, 0..4, 0..2]
    components.each_with_index do |field, index|
      value = values[index]
      errors << "#{path}: #{field} out of range #{ranges[index]}" if value && !ranges[index].cover?(value)
    end
    total = number(frontmatter["纸笔得分"])
    errors << "#{path}: 纸笔得分 must be numeric when score components are present" if total.nil?
    if total && values.all?
      expected = values.sum
      errors << "#{path}: 纸笔得分 #{total} does not equal component sum #{expected}" unless total == expected
    end
  end

  return unless %w[structure content all].include?(phase)
  errors << "#{path}: missing state control view" unless body.include?('dv.view("11_数据结构/系统/状态控件"')
  errors << "#{path}: missing question navigation view" unless body.include?('dv.view("11_数据结构/系统/题目导航"')
  errors << "#{path}: contains duplicated legacy status marker" if body.include?("const statuses = ['未开始'")
  errors << "#{path}: contains static status conflict" if body.include?("当前状态：**未开始**")
  return unless %w[content all].include?(phase)
  %w[独立作答 训练控制台 错因复盘].each { |label| errors << "#{path}: missing content label #{label}" unless body.include?(label) }
  if frontmatter["来源"] == "408真题"
    compact = body.gsub(/\s+/, "")
    %w[算法思想 正确性说明 C/C++代码 复杂度分析 评分点].each { |label| errors << "#{path}: missing 408 content label #{label}" unless compact.include?(label) }
    errors << "#{path}: missing success callout [!success]-" unless body.include?("[!success]-")
  elsif frontmatter["来源"] == "LeetCode"
    errors << "#{path}: missing LeetCode label 我的实现与复盘" unless body.include?("我的实现与复盘")
  end
end

def validate_pattern(path, frontmatter, body, phase, errors)
  PATTERN_FIELDS.each { |field| errors << "#{path}: missing pattern field #{field}" unless frontmatter.key?(field) }
  errors << "#{path}: cssclasses must contain ds-408-pattern" unless Array(frontmatter["cssclasses"]).include?("ds-408-pattern")
  return unless %w[content all].include?(phase)
  %w[识别信号 核心不变量 C/C++纸笔模板 正确性说明 复杂度 常见边界与失分点 可迁移变化 模板闭卷复写].each do |label|
    errors << "#{path}: missing pattern content label #{label}" unless body.include?(label)
  end
  errors << "#{path}: contains legacy handwriting placeholder" if body.include?("在此手写核心模板代码")
end

phase = selected_phase
errors = []
problem_paths = (PROJECT_ROOT.join("刷题记录").glob("*.md")).sort
pattern_paths = (PROJECT_ROOT.join("母题模板").glob("*.md")).sort
problems = problem_paths.map { |path| [path, *parse_note(path, errors)] }
patterns = pattern_paths.map { |path| [path, *parse_note(path, errors)] }
errors << "problem count is #{problems.length}; expected 57" unless problems.length == 57
errors << "pattern count is #{patterns.length}; expected 12" unless patterns.length == 12

problems.each { |path, fm, body| validate_problem(path, fm, body, phase, errors) }
patterns.each { |path, fm, body| validate_pattern(path, fm, body, phase, errors) }

exam_count = problems.count { |_path, fm, _body| fm["来源"] == "408真题" }
errors << "exam problem count is #{exam_count}; expected 10" unless exam_count == 10
pending_count = problems.count { |_path, fm, _body| fm["证据状态"] == "待校准" }
errors << "pending evidence count is #{pending_count}; expected range 0..15" unless (0..15).cover?(pending_count)

unless %w[structure content all].include?(phase)
  # Data validation intentionally does not require migrated dashboard pages.
else
  required_pages = ["408作战台.md", "总览.md", "错题回炉.md", "训练数据校准.md", "真题/真题冲刺.md"]
  required_pages.each do |relative|
    path = PROJECT_ROOT.join(relative)
    errors << "missing required page #{path}" unless path.file?
  end
  overview = PROJECT_ROOT.join("总览.md")
  if overview.file?
    text = overview.read
    %w[系统/统计卡 系统/复习队列 系统/题库表].each { |label| errors << "#{overview}: missing component view path #{label}" unless text.include?(label) }
  end
  dashboard = PROJECT_ROOT.join("408作战台.md")
  if dashboard.file?
    text = dashboard.read
    errors << "#{dashboard}: missing dashboard: true" unless text.include?("dashboard: true")
    errors << "#{dashboard}: missing type: dataview" unless text.include?("type: dataview")
  end
end

if errors.empty?
  puts "PASS phase=#{phase} problems=#{problems.length} exams=#{exam_count} patterns=#{patterns.length} pending=#{pending_count}"
  exit 0
end
errors.each { |error| warn "ERROR #{error}" }
warn "FAIL phase=#{phase} errors=#{errors.length}"
exit 1
