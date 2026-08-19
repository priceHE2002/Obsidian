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
end
