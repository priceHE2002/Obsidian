#!/usr/bin/env ruby
# frozen_string_literal: true
require "minitest/autorun"
require "tmpdir"
require "stringio"
require "fileutils"
require "pathname"
require "yaml"
require_relative "../脚本/validate-training-vault"

class ValidatorRegressionTest < Minitest::Test
  def setup
    @dir = Dir.mktmpdir("vault-validator-")
    @root = Pathname(@dir)
    (@root / "刷题记录").mkpath
    (@root / "母题模板").mkpath
    57.times { |i| write_problem(i + 1, i < 10 ? "408真题" : "LeetCode") }
    12.times { |i| write_pattern(i + 1) }
  end

  def teardown
    FileUtils.remove_entry(@dir)
  end

  def execute(phase = "data")
    out = StringIO.new
    err = StringIO.new
    rc = TrainingVaultValidator.main(argv: [phase], root: @root, out: out, err: err)
    [rc, out.string, err.string]
  end

  def test_valid_data_phase_passes_and_flag_phase_is_supported
    rc, out, err = execute
    assert_equal 0, rc
    assert_equal "PASS phase=data problems=57 exams=10 patterns=12 pending=0\n", out
    assert_empty err
    rc, out, = execute("--phase=data")
    assert_equal 0, rc
    assert_includes out, "PASS phase=data"
  end

  def test_malformed_yaml_and_invalid_utf8_are_reported_without_raising
    File.write(@root / "刷题记录" / "001.md", "---\n: bad: [\n---\n", encoding: "UTF-8")
    File.binwrite(@root / "刷题记录" / "002.md", "---\n\xFF\n---\n")
    rc, = execute
    refute_equal 0, rc
    _, _, err = execute
    assert_includes err, "invalid YAML frontmatter"
    assert_includes err, "not valid UTF-8"
  end

  def test_source_typo_and_partial_score_are_rejected
    path = @root / "刷题记录" / "001.md"
    text = path.read.gsub("来源: 408真题", "来源: 408真题X").gsub("思想得分: ''", "思想得分: 0.1")
    path.write(text)
    rc, _, err = execute
    refute_equal 0, rc
    assert_includes err, "来源 must be exactly"
    assert_includes err, "all five score fields"
  end

  def test_decimal_score_sum_is_accepted
    path = @root / "刷题记录" / "001.md"
    text = path.read
    text = text.sub("思想得分: ''", "思想得分: 0.1").sub("正确性得分: ''", "正确性得分: 0.2").sub("代码得分: ''", "代码得分: 0.1").sub("复杂度得分: ''", "复杂度得分: 0.2").sub("纸笔得分: ''", "纸笔得分: 0.6")
    path.write(text)
    rc, out, err = execute
    assert_equal 0, rc, err
    assert_includes out, "PASS phase=data"
  end

  def test_invalid_date_string_and_array_are_rejected
    path = @root / "刷题记录" / "001.md"
    text = path.read.sub("首刷日期: ''", "首刷日期: 2024/01/01").sub("二刷日期: ''", "二刷日期: [2024-01-01]")
    path.write(text)
    rc, _, err = execute
    refute_equal 0, rc
    assert_includes err, "首刷日期 must be"
    assert_includes err, "二刷日期 must be scalar"
  end

  def test_structure_phase_requires_dashboard_page_and_metadata
    ( @root / "总览.md").write("---\n---\n系统/统计卡 系统/复习队列 系统/题库表")
    ( @root / "错题回炉.md").write("")
    ( @root / "训练数据校准.md").write("")
    ( @root / "真题").mkpath
    ( @root / "真题" / "真题冲刺.md").write("")
    rc, _, err = execute("structure")
    refute_equal 0, rc
    assert_includes err, "missing required page"
    assert_includes err, "408作战台.md"
  end

  private

  def write_problem(number, source)
    fm = {
      "题号" => number, "母题" => "M01", "母题模块" => "模块", "章节" => "章节", "来源" => source,
      "引用" => "引用", "标题" => "标题", "优先级" => "A", "限时" => 30, "手写" => "建议", "训练目标" => "目标",
      "状态" => "未开始", "首刷日期" => "", "首刷耗时" => "", "首刷结果" => "未完成", "二刷日期" => "", "二刷结果" => "未完成",
      "纸笔日期" => "", "纸笔得分" => "", "错误类型" => "无", "下次复习" => "", "复习次数" => 0, "掌握评分" => "",
      "薄弱点" => "", "备注" => "", "证据状态" => "有效", "最后训练" => "", "思想得分" => "", "正确性得分" => "", "代码得分" => "", "复杂度得分" => "",
      "cssclasses" => ["ds-408-problem"]
    }
    (@root / "刷题记录" / format("%03d.md", number)).write("---\n#{fm.to_yaml.sub(/\A---\n/, "")}---\n")
  end

  def write_pattern(number)
    fm = { "模板状态" => "有效", "上次复写" => "", "下次复写" => "", "复写得分" => "", "cssclasses" => ["ds-408-pattern"] }
    (@root / "母题模板" / format("M%02d.md", number)).write("---\n#{fm.to_yaml.sub(/\A---\n/, "")}---\n")
  end
end
