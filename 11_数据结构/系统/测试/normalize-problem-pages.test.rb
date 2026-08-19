#!/usr/bin/env ruby
# frozen_string_literal: true

require "minitest/autorun"
require "tmpdir"
require "fileutils"
require "stringio"

require_relative "../脚本/normalize-problem-pages"

class NormalizeProblemPagesTest < Minitest::Test
  def with_fixture(files)
    Dir.mktmpdir("normalize-pages") do |root|
      directory = File.join(root, "刷题记录")
      FileUtils.mkdir_p(directory)
      files.each { |name, contents| File.binwrite(File.join(directory, name), contents) }
      yield root, directory
    end
  end

  def legacy(source: "LeetCode", ending: "\n", note: "（记录算法思想、关键代码、复杂度分析、易错点）")
    [
      "---", "来源: #{source}", "错误类型: 无", "薄弱点: ''", "---", "",
      "# 标题", "", "题面内容", "", "## ⚡ 状态快速切换", "旧状态", "",
      "## 🎯 训练目标", "保持目标", "", "## 📊 状态追踪", "旧追踪", "",
      "## 📝 解题笔记", note, "", "## 📚 错题复盘",
      "- **错误类型**：无", "- **薄弱点**：—", "- **复习次数**：0，**下次复习**：—", "- **备注**：—", ""
    ].join(ending)
  end

  def run_normalizer(root, **options)
    out, err = StringIO.new, StringIO.new
    status = NormalizeProblemPages.main(argv: ["--root", root, "--fixture"], out: out, err: err, **options)
    [status, out.string, err.string]
  end

  def read_utf8(path)
    File.binread(path).force_encoding(Encoding::UTF_8)
  end

  def test_leetcode_preserves_frontmatter_prefix_goal_and_notes_then_is_idempotent
    original = legacy.sub("## ⚡ 状态快速切换", "\n## ⚡ 状态快速切换")
    with_fixture("a.md" => original) do |root, directory|
      File.chmod(0o600, File.join(directory, "a.md"))
      status, out, err = run_normalizer(root)
      assert_equal 0, status, err
      assert_equal "normalized=1 scanned=1 leetcode=1 exams=0\n", out
      normalized = read_utf8(File.join(directory, "a.md"))
      assert_equal original[/\A.*?---\n/m], normalized[/\A.*?---\n/m]
      assert_includes normalized, "# 标题\n\n题面内容\n\n\n## 🎯 训练目标\n保持目标\n"
      assert_includes normalized, "## 📝 我的实现与复盘\n（记录算法思想、关键代码、复杂度分析、易错点）\n"
      assert_equal 1, normalized.scan('dv.view("11_数据结构/系统/状态控件"').length
      assert_equal 1, normalized.scan('dv.view("11_数据结构/系统/题目导航"').length
      refute_includes normalized, "状态快速切换"
      assert_equal 0o600, File.stat(File.join(directory, "a.md")).mode & 0o777
      status, out, err = run_normalizer(root)
      assert_equal 0, status, err
      assert_equal "normalized=0 scanned=1 leetcode=1 exams=0\n", out
      assert_equal normalized, read_utf8(File.join(directory, "a.md"))
    end
  end

  def test_exam_quotes_every_note_line_in_collapsed_answer
    notes = "**算法思想**：保留 2018 答案。\n\n```cpp\nint f() { return 1; }\n```"
    with_fixture("exam.md" => legacy(source: "408真题", note: notes)) do |root, directory|
      status, = run_normalizer(root)
      assert_equal 0, status
      body = read_utf8(File.join(directory, "exam.md"))
      assert_includes body, "## 🧠 满分参考答案\n> [!success]- 展开满分参考答案（完成闭卷作答后再看）\n> **算法思想**：保留 2018 答案。\n>\n> ```cpp\n> int f() { return 1; }\n> ```"
    end
  end

  def test_invalid_file_aborts_entire_batch_without_writes
    good = legacy
    bad = legacy.sub("## 📊 状态追踪", "## 🎯 训练目标")
    with_fixture("good.md" => good, "bad.md" => bad) do |root, directory|
      status, _out, err = run_normalizer(root)
      assert_equal 1, status
      assert_match(/duplicate legacy boundary/, err)
      assert_equal good.b, File.binread(File.join(directory, "good.md"))
      assert_equal bad.b, File.binread(File.join(directory, "bad.md"))
    end
  end

  def test_rejects_unclassified_review_content_and_preserves_crlf
    bad = legacy(ending: "\r\n").sub("- **备注**：—\r\n", "- **备注**：—\r\n用户内容\r\n")
    with_fixture("bad.md" => bad) do |root, directory|
      status, _out, err = run_normalizer(root)
      assert_equal 1, status
      assert_match(/unclassified content after/, err)
      assert_equal bad.b, File.binread(File.join(directory, "bad.md"))
    end
  end

  def test_rejects_invalid_utf8_yaml_and_requires_57_in_production_mode
    with_fixture("bad.md" => "---\n来源: LeetCode\n---\n\xFF") do |root, directory|
      out, err = StringIO.new, StringIO.new
      assert_equal 1, NormalizeProblemPages.main(argv: ["--root", root, "--fixture"], out: out, err: err)
      assert_match(/valid UTF-8/, err.string)
      File.binwrite(File.join(directory, "bad.md"), legacy)
      assert_equal 1, NormalizeProblemPages.main(argv: ["--root", root], out: out, err: err)
      assert_match(/expected 57/, err.string)
    end
  end

  def test_production_count_accepts_47_leetcode_and_10_exam_pages
    files = {}
    47.times { |index| files[format("lc-%02d.md", index)] = legacy }
    10.times { |index| files[format("exam-%02d.md", index)] = legacy(source: "408真题") }
    with_fixture(files) do |root, _directory|
      out, err = StringIO.new, StringIO.new
      status = NormalizeProblemPages.main(argv: ["--root", root], out: out, err: err)
      assert_equal 0, status, err.string
      assert_equal "normalized=57 scanned=57 leetcode=47 exams=10\n", out.string
    end
  end

  def test_all_prefix_terminal_newline_counts_are_byte_stable_after_first_write
    (1..4).each do |count|
      original = legacy.sub("## ⚡ 状态快速切换", ("\n" * (count - 1)) + "## ⚡ 状态快速切换")
      with_fixture("#{count}.md" => original) do |root, directory|
        assert_equal 0, run_normalizer(root).first
        first = File.binread(File.join(directory, "#{count}.md"))
        assert_equal 0, run_normalizer(root).first
        assert_equal first, File.binread(File.join(directory, "#{count}.md"))
      end
    end
  end

  def test_rejects_normalized_page_missing_one_independent_answer_item
    with_fixture("a.md" => legacy) do |root, directory|
      assert_equal 0, run_normalizer(root).first
      path = File.join(directory, "a.md")
      broken = read_utf8(path).sub("- 空间复杂度及依据：\n", "").b
      File.binwrite(path, broken)
      status, _out, err = run_normalizer(root)
      assert_equal 1, status
      assert_match(/missing legacy boundary/, err)
      assert_equal broken, File.binread(path)
    end
  end

  def test_rejects_normalized_page_with_tail_or_duplicate_view
    with_fixture("a.md" => legacy) do |root, directory|
      assert_equal 0, run_normalizer(root).first
      path = File.join(directory, "a.md")
      tail = File.binread(path) + "未分类尾巴\n".b
      File.binwrite(path, tail)
      assert_equal 1, run_normalizer(root).first
      File.binwrite(path, read_utf8(path).sub("未分类尾巴\n", "") + "\n```dataviewjs\n#{NormalizeProblemPages::PROBLEM_VIEW}\n```\n")
      status, _out, err = run_normalizer(root)
      assert_equal 1, status
      assert_match(/missing legacy boundary/, err)
    end
  end

  class SecondStageFailure < NormalizeProblemPages::FileOperations
    def initialize
      @writes = 0
    end

    def write_stage(path, contents, mode)
      super
      @writes += 1
      raise Errno::EIO, "second stage" if @writes == 2
    end
  end

  class SecondReplaceFailure < NormalizeProblemPages::FileOperations
    def initialize
      @replacements = 0
    end

    def rename(from, to)
      if from.to_s.include?(".normalize-stage-") && to.to_s.end_with?(".md")
        @replacements += 1
        raise Errno::EIO, "second replacement" if @replacements == 2
      end
      super
    end
  end

  class TransientBackupCleanupFailure < NormalizeProblemPages::FileOperations
    def initialize
      @failed = false
    end

    def delete(path)
      if path.include?(".normalize-backup-") && !@failed
        @failed = true
        raise Errno::EIO, "transient backup cleanup"
      end
      super
    end
  end

  def test_second_stage_failure_leaves_no_artifacts_and_preserves_bytes_and_mode
    original = legacy
    with_fixture("a.md" => original, "b.md" => original) do |root, directory|
      paths = %w[a.md b.md].map { |name| File.join(directory, name) }
      paths.each { |path| File.chmod(0o600, path) }
      before = paths.map { |path| [File.binread(path), File.stat(path).mode & 0o777] }
      status, _out, err = run_normalizer(root, file_operations: SecondStageFailure.new)
      assert_equal 1, status
      assert_match(/second stage/, err)
      assert_equal before, paths.map { |path| [File.binread(path), File.stat(path).mode & 0o777] }
      assert_empty Dir.glob(File.join(directory, "*.normalize-*"))
    end
  end

  def test_second_replace_failure_rolls_back_all_bytes_and_modes_without_artifacts
    with_fixture("a.md" => legacy, "b.md" => legacy) do |root, directory|
      paths = %w[a.md b.md].map { |name| File.join(directory, name) }
      File.chmod(0o600, paths[0])
      File.chmod(0o640, paths[1])
      before = paths.map { |path| [File.binread(path), File.stat(path).mode & 0o777] }
      status, _out, err = run_normalizer(root, file_operations: SecondReplaceFailure.new)
      assert_equal 1, status
      assert_match(/second replacement/, err)
      assert_equal before, paths.map { |path| [File.binread(path), File.stat(path).mode & 0o777] }
      assert_empty Dir.glob(File.join(directory, "*.normalize-*"))
    end
  end

  def test_transient_backup_cleanup_keeps_committed_files_and_retries_artifact_removal
    with_fixture("a.md" => legacy, "b.md" => legacy) do |root, directory|
      status, out, err = run_normalizer(root, file_operations: TransientBackupCleanupFailure.new)
      assert_equal 0, status, err
      assert_equal "normalized=2 scanned=2 leetcode=2 exams=0\n", out
      assert File.file?(File.join(directory, "a.md"))
      assert File.file?(File.join(directory, "b.md"))
      assert_empty Dir.glob(File.join(directory, "*.normalize-*"))
    end
  end

  def test_rejects_invalid_and_duplicate_yaml_before_any_batch_write
    malformed = "---\n来源: [\n---\n"
    duplicate = "---\n来源: LeetCode\n来源: LeetCode\n---\n"
    [malformed, duplicate].each do |bad|
      with_fixture("good.md" => legacy, "bad.md" => bad) do |root, directory|
        good = File.binread(File.join(directory, "good.md"))
        status, _out, err = run_normalizer(root)
        assert_equal 1, status
        assert_match(/invalid YAML|duplicate YAML/, err)
        assert_equal good, File.binread(File.join(directory, "good.md"))
      end
    end
  end

  def test_rejects_each_missing_duplicate_and_out_of_order_legacy_boundary
    NormalizeProblemPages::LEGACY_HEADINGS.each do |heading|
      bad = legacy.sub("#{heading}\n", "")
      with_fixture("bad.md" => bad) do |root, _directory|
        status, _out, err = run_normalizer(root)
        assert_equal 1, status
        assert_match(/missing legacy boundary/, err)
      end
    end
    duplicate = legacy.sub("## 📊 状态追踪", "## 🎯 训练目标")
    out_of_order = legacy.sub("## 🎯 训练目标", "## TEMPORARY").sub("## 📊 状态追踪", "## 🎯 训练目标").sub("## TEMPORARY", "## 📊 状态追踪")
    [duplicate, out_of_order].each do |bad|
      with_fixture("bad.md" => bad) do |root, _directory|
        status, _out, err = run_normalizer(root)
        assert_equal 1, status
        assert_match(/duplicate legacy boundary|out of order/, err)
      end
    end
  end

  def test_crlf_leetcode_fence_is_preserved_and_second_run_is_byte_identical
    notes = "第一段\r\n\r\n```cpp\r\nint f() { return 1; }\r\n```"
    with_fixture("a.md" => legacy(ending: "\r\n", note: notes)) do |root, directory|
      assert_equal 0, run_normalizer(root).first
      path = File.join(directory, "a.md")
      first = File.binread(path)
      assert_includes first, "> [!question] 闭卷作答区\r\n".b
      assert_includes first, notes.b
      assert_equal 0, run_normalizer(root).first
      assert_equal first, File.binread(path)
    end
  end

  def test_rejects_normalized_source_mismatch_missing_review_items_and_duplicate_source_heading
    with_fixture("a.md" => legacy) do |root, directory|
      assert_equal 0, run_normalizer(root).first
      path = File.join(directory, "a.md")
      cases = [
        read_utf8(path).sub("- 下一步：根据控制台下次复习日期闭卷重做。\n", ""),
        read_utf8(path).sub("## 📝 我的实现与复盘", "## 🧠 满分参考答案"),
        read_utf8(path).sub("## 🔁 错因复盘", "## 📝 我的实现与复盘\n\n## 🔁 错因复盘")
      ]
      cases.each do |broken|
        File.binwrite(path, broken)
        status, _out, err = run_normalizer(root)
        assert_equal 1, status
        assert_match(/missing legacy boundary/, err)
      end
    end
  end

  def test_leetcode_unknown_h2_and_fenced_h2_in_goal_and_notes_survive_second_run
    note = "第一段\n\n## 自定义分析\n\n```cpp\n## 不是页面章节\n```\n\n末段"
    page = legacy(note: note).sub("保持目标", "保持目标\n\n## 目标子标题")
    with_fixture("a.md" => page) do |root, directory|
      assert_equal 0, run_normalizer(root).first
      path = File.join(directory, "a.md")
      first = File.binread(path)
      assert_includes first, "## 目标子标题\n".b
      assert_includes first, note.b
      assert_equal 0, run_normalizer(root).first
      assert_equal first, File.binread(path)
    end
  end

  def test_exam_unknown_h2_and_fenced_h2_are_quoted_and_idempotent
    note = "2018 答案\n\n## 自定义分析\n\n```cpp\n## 代码内标题\n```"
    with_fixture("a.md" => legacy(source: "408真题", note: note)) do |root, directory|
      assert_equal 0, run_normalizer(root).first
      path = File.join(directory, "a.md")
      first = File.binread(path)
      assert_includes first, "> ## 自定义分析\n>\n> ```cpp\n> ## 代码内标题\n> ```".b
      assert_equal 0, run_normalizer(root).first
      assert_equal first, File.binread(path)
    end
  end

  def test_unknown_h2_after_navigation_is_unclassified_tail_not_protected_note_content
    with_fixture("a.md" => legacy) do |root, directory|
      assert_equal 0, run_normalizer(root).first
      path = File.join(directory, "a.md")
      broken = File.binread(path) + "## 导航后的未知标题\n".b
      File.binwrite(path, broken)
      status, _out, err = run_normalizer(root)
      assert_equal 1, status
      assert_match(/missing legacy boundary/, err)
      assert_equal broken, File.binread(path)
    end
  end

  def test_lc_fence_hides_all_known_page_headings_on_both_runs
    headings = (NormalizeProblemPages::LEGACY_HEADINGS + ["## ✍️ 独立作答", "## 🧭 训练控制台", "## 🧭 导航"]).join("\n")
    note = "正文\n\n```md\n#{headings}\n```\n\n结束"
    with_fixture("a.md" => legacy(note: note)) do |root, directory|
      assert_equal 0, run_normalizer(root).first
      path = File.join(directory, "a.md")
      first = File.binread(path)
      assert_includes first, headings.b
      assert_equal 0, run_normalizer(root).first
      assert_equal first, File.binread(path)
    end
  end

  def test_fence_closes_only_with_matching_character_and_sufficient_length
    cases = [
      "````md\n```\n## 📚 错题复盘\n````",
      "~~~~md\n~~~\n## 📚 错题复盘\n~~~~",
      "```md\n~~~\n## 📚 错题复盘\n```"
    ]
    cases.each_with_index do |fence, index|
      with_fixture("#{index}.md" => legacy(note: "前言\n\n#{fence}\n\n后记")) do |root, directory|
        assert_equal 0, run_normalizer(root).first
        path = File.join(directory, "#{index}.md")
        first = File.binread(path)
        assert_includes first, fence.b
        assert_equal 0, run_normalizer(root).first
        assert_equal first, File.binread(path)
      end
    end
  end

  def test_unclosed_legacy_fence_is_rejected_without_writes
    bad = legacy(note: "安全保留\n```md\n## 📚 错题复盘")
    with_fixture("good.md" => legacy, "bad.md" => bad) do |root, directory|
      original = File.binread(File.join(directory, "good.md"))
      status, _out, err = run_normalizer(root)
      assert_equal 1, status
      assert_match(/unclosed Markdown fence/, err)
      assert_equal original, File.binread(File.join(directory, "good.md"))
    end
  end
end
