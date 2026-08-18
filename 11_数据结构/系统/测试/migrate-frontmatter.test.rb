#!/usr/bin/env ruby
# frozen_string_literal: true

require "minitest/autorun"
require "tmpdir"
require "fileutils"
require "yaml"
require "date"
require "shellwords"

class MigrateFrontmatterTest < Minitest::Test
  SCRIPT = File.expand_path("../脚本/migrate-frontmatter.rb", __dir__)

  def with_vault
    Dir.mktmpdir("migrate-frontmatter") do |root|
      FileUtils.mkdir_p(File.join(root, "刷题记录"))
      FileUtils.mkdir_p(File.join(root, "母题模板"))
      yield root
    end
  end

  def write_note(root, kind, name, frontmatter, body = "# 保持这个正文\n二进制: \xE2\x9C\x93\n")
    path = File.join(root, kind, name)
    File.binwrite(path, "---\n#{frontmatter}---\n#{body}")
    path
  end

  def run_migration(root)
    output = `ruby #{Shellwords.escape(SCRIPT)} --root #{Shellwords.escape(root)} 2>&1`
    [output, $?.exitstatus]
  end

  def frontmatter_and_body(path)
    raw = File.binread(path)
    match = raw.match(/\A---\n(.*?)\n---\n(.*)\z/m)
    [YAML.safe_load(match[1], permitted_classes: [Date], aliases: false), match[2]]
  end

  def test_problem_migration_preserves_body_existing_values_and_sets_pending_evidence
    with_vault do |root|
      path = write_note(root, "刷题记录", "001.md", <<~YAML)
        状态: 已AC
        首刷日期: ""
        首刷结果: 未完成
        思想得分: 9
        cssclasses: old-class
        自定义字段: 保留
      YAML
      original_body = File.binread(path).split("---\n", 3).last

      output, status = run_migration(root)

      assert_equal 0, status, output
      assert_includes output, "changed=1"
      frontmatter, body = frontmatter_and_body(path)
      assert_equal original_body, body
      assert_equal "已AC", frontmatter["状态"]
      assert_equal "", frontmatter["首刷日期"]
      assert_equal "未完成", frontmatter["首刷结果"]
      assert_equal 9, frontmatter["思想得分"]
      assert_equal "保留", frontmatter["自定义字段"]
      assert_equal ["old-class", "ds-408-problem"], frontmatter["cssclasses"]
      assert_equal "待校准", frontmatter["证据状态"]
      %w[最后训练 正确性得分 代码得分 复杂度得分].each { |key| assert_equal "", frontmatter[key] }
    end
  end

  def test_evidence_status_mapping_for_each_status
    cases = {
      "未开始" => ["", "", "", "", "", "", "有效"],
      "学习中" => ["", "", "", "", "", "", "有效"],
      "已AC" => ["2026-01-01", "独立通过", "", "", "", "", "有效"],
      "已二刷" => ["", "", "2026-01-02", "独立通过", "", "", "有效"],
      "可纸笔" => ["", "", "", "", "2026-01-03", "8", "有效"],
      "可纸笔待校准" => ["", "", "", "", "", "7.9", "待校准"]
    }
    with_vault do |root|
      cases.each_with_index do |(label, values), index|
        status = label == "可纸笔待校准" ? "可纸笔" : label
        write_note(root, "刷题记录", format("%03d.md", index), <<~YAML)
          状态: #{status}
          首刷日期: "#{values[0]}"
          首刷结果: "#{values[1]}"
          二刷日期: "#{values[2]}"
          二刷结果: "#{values[3]}"
          纸笔日期: "#{values[4]}"
          纸笔得分: "#{values[5]}"
        YAML
      end

      output, status = run_migration(root)
      assert_equal 0, status, output
      cases.each_with_index do |(_label, values), index|
        frontmatter, = frontmatter_and_body(File.join(root, "刷题记录", format("%03d.md", index)))
        assert_equal values[6], frontmatter["证据状态"]
      end
    end
  end

  def test_array_cssclasses_mother_defaults_and_idempotence
    with_vault do |root|
      problem = write_note(root, "刷题记录", "001.md", "状态: 未开始\ncssclasses:\n  - existing\n")
      mother = write_note(root, "母题模板", "M01.md", "cssclasses:\n  - existing\n模板状态: 已完成\n")

      output, status = run_migration(root)
      assert_equal 0, status, output
      problem_fm, = frontmatter_and_body(problem)
      mother_fm, = frontmatter_and_body(mother)
      assert_equal ["existing", "ds-408-problem"], problem_fm["cssclasses"]
      assert_equal ["existing", "ds-408-pattern"], mother_fm["cssclasses"]
      assert_equal "已完成", mother_fm["模板状态"]
      %w[上次复写 下次复写 复写得分].each { |key| assert_equal "", mother_fm[key] }

      output, status = run_migration(root)
      assert_equal 0, status, output
      assert_includes output, "changed=0"
    end
  end

  def test_absent_cssclasses_are_persisted_for_problems_and_mothers_and_are_idempotent
    with_vault do |root|
      problem = write_note(root, "刷题记录", "001.md", "状态: 未开始\n")
      mother = write_note(root, "母题模板", "M01.md", "母题编号: M01\n")

      output, status = run_migration(root)

      assert_equal 0, status, output
      assert_includes output, "changed=2"
      assert_includes output, "scanned=2"
      problem_fm, = frontmatter_and_body(problem)
      mother_fm, = frontmatter_and_body(mother)
      assert_equal ["ds-408-problem"], problem_fm["cssclasses"]
      assert_equal "有效", problem_fm["证据状态"]
      %w[最后训练 思想得分 正确性得分 代码得分 复杂度得分].each { |key| assert_equal "", problem_fm[key] }
      assert_equal ["ds-408-pattern"], mother_fm["cssclasses"]
      assert_equal "未开始", mother_fm["模板状态"]
      %w[上次复写 下次复写 复写得分].each { |key| assert_equal "", mother_fm[key] }

      output, status = run_migration(root)
      assert_equal 0, status, output
      assert_includes output, "changed=0"
      assert_includes output, "scanned=2"
    end
  end

  def test_malformed_input_aborts_without_writing_any_file
    with_vault do |root|
      valid = write_note(root, "刷题记录", "001.md", "状态: 未开始\n")
      malformed = File.join(root, "母题模板", "bad.md")
      File.binwrite(malformed, "---\ncssclasses: [unterminated\n---\nbody\n")
      original = File.binread(valid)

      output, status = run_migration(root)

      refute_equal 0, status
      assert_includes output, malformed
      assert_equal original, File.binread(valid)
    end
  end
end
