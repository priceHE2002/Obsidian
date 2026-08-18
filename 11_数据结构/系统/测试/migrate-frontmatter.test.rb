#!/usr/bin/env ruby
# frozen_string_literal: true

require "minitest/autorun"
require "tmpdir"
require "fileutils"
require "yaml"
require "date"
require "shellwords"
require "stringio"

MIGRATION_SCRIPT = File.expand_path("../脚本/migrate-frontmatter.rb", __dir__)
require MIGRATION_SCRIPT

class MigrateFrontmatterTest < Minitest::Test
  SCRIPT = MIGRATION_SCRIPT

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

  def run_runner(root, file_operations)
    output = StringIO.new
    error = StringIO.new
    status = TrainingFrontmatterMigration::Runner.new(
      root: root,
      out: output,
      err: error,
      file_operations: file_operations
    ).run
    ["#{output.string}#{error.string}", status]
  end

  def assert_no_transaction_artifacts(root)
    Dir[File.join(root, "**", "*")].each do |path|
      refute_match(/\.migrate-frontmatter-|\.backup\z/, File.basename(path), "left artifact: #{path}")
    end
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
      first_problem_bytes = File.binread(problem)
      first_mother_bytes = File.binread(mother)

      output, status = run_migration(root)
      assert_equal 0, status, output
      assert_includes output, "changed=0"
      assert_includes output, "scanned=2"
      assert_equal first_problem_bytes, File.binread(problem)
      assert_equal first_mother_bytes, File.binread(mother)
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

  def test_duplicate_yaml_key_aborts_whole_batch_without_writes
    with_vault do |root|
      valid = write_note(root, "刷题记录", "001.md", "状态: 未开始\n")
      duplicate = write_note(root, "母题模板", "M01.md", "模板状态: 未开始\n模板状态: 已完成\n")
      original = File.binread(valid)

      output, status = run_migration(root)

      refute_equal 0, status
      assert_includes output, duplicate
      assert_match(/duplicate YAML key/, output)
      assert_equal original, File.binread(valid)
    end
  end

  def test_existing_evidence_and_user_fields_are_never_overwritten
    with_vault do |root|
      path = write_note(root, "刷题记录", "001.md", <<~YAML)
        状态: 未知状态
        证据状态: 人工确认
        最后训练: "2026-08-01"
        思想得分: 10
        正确性得分: 9
        代码得分: 8
        复杂度得分: 7
      YAML

      output, status = run_migration(root)

      assert_equal 0, status, output
      frontmatter, = frontmatter_and_body(path)
      assert_equal "人工确认", frontmatter["证据状态"]
      assert_equal "2026-08-01", frontmatter["最后训练"]
      assert_equal 10, frontmatter["思想得分"]
      assert_equal 9, frontmatter["正确性得分"]
      assert_equal 8, frontmatter["代码得分"]
      assert_equal 7, frontmatter["复杂度得分"]
    end
  end

  def test_ac_correction_is_valid_and_unknown_status_is_pending
    with_vault do |root|
      corrected = write_note(root, "刷题记录", "001.md", "状态: 已AC\n首刷日期: 2026-08-01\n首刷结果: 订正通过\n")
      unknown = write_note(root, "刷题记录", "002.md", "状态: 外部状态\n")

      output, status = run_migration(root)

      assert_equal 0, status, output
      corrected_fm, = frontmatter_and_body(corrected)
      unknown_fm, = frontmatter_and_body(unknown)
      assert_equal "有效", corrected_fm["证据状态"]
      assert_equal "待校准", unknown_fm["证据状态"]
    end
  end

  def test_crlf_body_is_preserved_byte_for_byte
    with_vault do |root|
      path = File.join(root, "刷题记录", "001.md")
      body = "\r\n# CRLF 正文\r\n字节不变\r\n"
      original = "---\r\n状态: 未开始\r\n---\r\n#{body}"
      File.binwrite(path, original)

      output, status = run_migration(root)

      assert_equal 0, status, output
      assert_equal body.b, File.binread(path).match(/\A---\r\n.*?\n---\r\n(.*)\z/m)[1]
    end
  end

  def test_failed_staging_leaves_all_notes_unchanged_and_no_artifacts
    with_vault do |root|
      problem = write_note(root, "刷题记录", "001.md", "状态: 未开始\n")
      mother = write_note(root, "母题模板", "M01.md", "母题编号: M01\n")
      originals = [problem, mother].to_h { |path| [path, File.binread(path)] }
      operations = FailingFileOperations.new(fail_stage_number: 2)

      output, status = run_runner(root, operations)

      refute_equal 0, status
      assert_includes output, "injected staging failure"
      originals.each { |path, bytes| assert_equal bytes, File.binread(path) }
      assert_no_transaction_artifacts(root)
    end
  end

  def test_failed_second_replacement_rolls_back_all_notes_and_no_artifacts
    with_vault do |root|
      problem = write_note(root, "刷题记录", "001.md", "状态: 未开始\n")
      mother = write_note(root, "母题模板", "M01.md", "母题编号: M01\n")
      originals = [problem, mother].to_h { |path| [path, File.binread(path)] }
      operations = FailingFileOperations.new(fail_replacement_number: 2)

      output, status = run_runner(root, operations)

      refute_equal 0, status
      assert_includes output, "injected replacement failure"
      originals.each { |path, bytes| assert_equal bytes, File.binread(path) }
      assert_no_transaction_artifacts(root)
    end
  end

  class FailingFileOperations
    def initialize(fail_stage_number: nil, fail_replacement_number: nil)
      @fail_stage_number = fail_stage_number
      @fail_replacement_number = fail_replacement_number
      @stage_calls = 0
      @replacement_calls = 0
    end

    def write_stage(path, contents)
      @stage_calls += 1
      raise IOError, "injected staging failure" if @stage_calls == @fail_stage_number

      File.open(path, "wb") do |file|
        file.write(contents)
        file.flush
        file.fsync
      end
    end

    def rename(source, destination)
      if source.to_s.end_with?(".tmp") && !destination.to_s.end_with?(".tmp")
        @replacement_calls += 1
        raise IOError, "injected replacement failure" if @replacement_calls == @fail_replacement_number
      end
      File.rename(source, destination)
    end

    def delete(path)
      File.delete(path) if File.exist?(path)
    end

    def exist?(path)
      File.exist?(path)
    end
  end
end
