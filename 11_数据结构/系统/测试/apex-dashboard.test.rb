#!/usr/bin/env ruby
# frozen_string_literal: true

require "digest"
require "json"
require "minitest/autorun"
require "pathname"
require "yaml"

class ApexDashboardContractTest < Minitest::Test
  REPO_ROOT = Pathname.new(__dir__).join("../../..").cleanpath
  VAULT_ROOT = REPO_ROOT.join("11_数据结构")
  PLUGIN_ROOT = REPO_ROOT.join(".obsidian/plugins/apex-dashboard")
  DASHBOARD_PATH = VAULT_ROOT.join("408作战台.md")

  RELEASE_FILES = {
    "main.js" => "0d934a4a8628e7c50038c6d9efac0f096920608c146f531f9fc75335f065a5ce",
    "manifest.json" => "4f4332694f4bbc6cc8bc6e7a7dbdd672c0ebc1e9871703356caa342176ff0a8c",
    "styles.css" => "a87bd505e4a4de37c3270046fd3539e26a35bb759047e2257a899a0f99c9b815"
  }.freeze

  EXISTING_COMMUNITY_PLUGINS = %w[
    fast-text-color
    easy-typing-obsidian
    highlightr-plugin
    mousewheel-image-zoom
    obsidian-git
    execute-code
    obsidian-image-toolkit
    obsidian-mindmap-nextgen
    realclaudian
    obsidian-latex-suite
    dataview
  ].freeze

  ACTIONS = [
    ["总览", "layout-dashboard", "11_数据结构/总览.md"],
    ["真题冲刺", "timer", "11_数据结构/真题/真题冲刺.md"],
    ["错题回炉", "rotate-ccw", "11_数据结构/错题回炉.md"],
    ["训练数据校准", "badge-check", "11_数据结构/训练数据校准.md"],
    ["使用说明", "book-open-check", "11_数据结构/使用说明.md"]
  ].freeze

  COLUMN_NAMES = %w[
    今日首要任务
    到期复习
    今日真题
    模板复写
    复习告警
    本周得分链
    训练阶段分布
  ].freeze

  EXPECTED_SETTINGS = {
    "dashboardFile" => "11_数据结构/408作战台.md",
    "recentDocCount" => 5,
    "language" => "zh",
    "stylePreset" => "matcha",
    "widgetWeatherEnabled" => false,
    "widgetWeatherCity" => "Shanghai",
    "widgetWeatherLat" => 31.23,
    "widgetWeatherLon" => 121.47,
    "pomodoroEnabled" => true,
    "pomodoroWorkMinutes" => 25,
    "pomodoroShortBreakMinutes" => 5,
    "pomodoroLongBreakMinutes" => 15,
    "pomodoroLongBreakInterval" => 4,
    "pomodoroDailyGoal" => 8,
    "pomodoroAutoStartBreak" => true,
    "pomodoroSoundEnabled" => true,
    "widgetLunarEnabled" => false,
    "widgetYearProgressEnabled" => false,
    "widgetCalendarEnabled" => false,
    "calendarExcludeFolders" => [],
    "widgetOrder" => %w[weather lunar pomodoro reading countdown yearProgress calendar],
    "wereadApiKey" => "",
    "wereadImportPath" => "Weread/划线",
    "ticktickRegion" => "dida365",
    "ticktickCookie" => "",
    "ticktickCsrf" => "",
    "ticktickTimezone" => "Asia/Shanghai",
    "disableNotePopover" => false,
    "customColors" => {},
    "bgImage" => "",
    "bgDim" => 40,
    "bgBlur" => 0,
    "bgSize" => "cover",
    "surfaceOpacity" => nil,
    "glassBlur" => nil,
    "radiusScale" => nil,
    "quickNotesEnabled" => false,
    "quickNotePresets" => [],
    "quickCaptureEnabled" => false,
    "quickCaptureTarget" => "",
    "quickCaptureFolder" => "",
    "quickCaptureTemplate" => "",
    "pinnedNotes" => [],
    "quickCommands" => [],
    "quickDailyEnabled" => false,
    "quickNoteGuideShownVersion" => "",
    "dataviewGuideShownVersion" => "",
    "countdownEnabled" => false,
    "countdowns" => [],
    "mediaTags" => {},
    "readingEnabled" => false,
    "readingSoundEnabled" => true,
    "taskTemplates" => [],
    "memoSavePath" => "",
    "taskArchivePath" => "归档/已完成.md",
    "backupEnabled" => false,
    "backupPeriod" => "daily",
    "backupMaxCount" => 10
  }.freeze

  def setup
    @frontmatter, @body = parse_frontmatter(DASHBOARD_PATH)
  end

  def test_installs_exact_official_release_assets
    RELEASE_FILES.each do |filename, expected_sha|
      path = PLUGIN_ROOT.join(filename)
      assert path.file?, "missing official release asset #{path}"
      assert_equal expected_sha, Digest::SHA256.file(path).hexdigest
    end

    manifest = parse_json(PLUGIN_ROOT.join("manifest.json"))
    assert_equal "apex-dashboard", manifest.fetch("id")
    assert_equal "1.8.6", manifest.fetch("version")
    assert_equal "1.8.7", manifest.fetch("minAppVersion")
  end

  def test_enables_apex_once_without_losing_existing_plugins
    plugins = parse_json(REPO_ROOT.join(".obsidian/community-plugins.json"))
    assert_equal EXISTING_COMMUNITY_PLUGINS, plugins.reject { |id| id == "apex-dashboard" }
    assert_equal 1, plugins.count("apex-dashboard")
    assert_equal "apex-dashboard", plugins.last
  end

  def test_data_json_is_the_complete_approved_secret_free_settings_object
    settings = parse_json(PLUGIN_ROOT.join("data.json"))
    assert_equal EXPECTED_SETTINGS, settings
    %w[wereadApiKey ticktickCookie ticktickCsrf].each do |credential|
      assert_equal "", settings.fetch(credential), "#{credential} must not contain account data"
    end
    refute_match(/wrk-[A-Za-z0-9_-]+|(?:cookie|csrf)\s*[:=]\s*[^\s\"']+/i, JSON.generate(settings))
  end

  def test_dashboard_parses_to_exact_actions_and_apex_186_persisted_query_shape
    assert_equal true, @frontmatter.fetch("dashboard")
    assert_equal({ "quote" => "数据结构的每一分，都来自限时、手写与复盘。", "author" => "408 备考" }, @frontmatter.fetch("banner"))

    actions = @frontmatter.fetch("quickActions")
    assert_equal ACTIONS.map { |name, icon, target| { "name" => name, "icon" => icon, "type" => "file", "target" => target } }, actions
    actions.each do |action|
      assert action.fetch("target").end_with?(".md")
      assert REPO_ROOT.join(action.fetch("target")).file?, "missing quick-action target #{action.fetch("target")}"
    end

    columns = @frontmatter.fetch("columns")
    assert_equal COLUMN_NAMES, columns.map { |column| column.fetch("name") }
    columns.each do |column|
      assert_equal "dataview", column.fetch("type")
      config = column.fetch("dataview")
      assert_equal %w[query title], config.keys.sort
      assert_balanced_dql(config.fetch("query"))
      refute_empty config.fetch("title")
    end
  end

  def test_dql_queries_encode_the_approved_408_priority_rules
    queries = @frontmatter.fetch("columns").to_h { |column| [column.fetch("name"), column.fetch("dataview").fetch("query")] }
    problem_scope = 'FROM "11_数据结构/刷题记录"'
    pattern_scope = 'FROM "11_数据结构/母题模板"'

    ["今日首要任务", "到期复习", "今日真题", "复习告警", "本周得分链", "训练阶段分布"].each do |name|
      assert_includes queries.fetch(name), problem_scope
    end
    assert_includes queries.fetch("模板复写"), pattern_scope

    primary = queries.fetch("今日首要任务")
    %w[下次复习 证据状态 408真题 优先级 手写 状态 题号].each { |term| assert_includes primary, term }
    primary_sort = primary.split(" SORT ", 2).fetch(1)
    assert_order(primary_sort, '下次复习', '证据状态', '408真题', '优先级', '手写', 'choice(状态 = "未开始"', '题号')
    assert_match(/LIMIT\s+1\z/i, primary)

    due = queries.fetch("到期复习")
    assert_match(/下次复习.+today.+OR.+证据状态\s*=\s*"待校准"/i, due)
    assert_match(/LIMIT\s+1\z/i, due)

    exam = queries.fetch("今日真题")
    assert_match(/来源\s*=\s*"408真题"/i, exam)
    assert_match(/状态\s*!=\s*"可纸笔".+OR.+证据状态\s*=\s*"待校准"/i, exam)
    assert_match(/LIMIT\s+1\z/i, exam)

    pattern = queries.fetch("模板复写")
    assert_match(/下次复写.+today.+OR.+模板状态\s*!=\s*"可复写"/i, pattern)
    assert_match(/LIMIT\s+1\z/i, pattern)

    alert = queries.fetch("复习告警")
    assert_includes alert, 'date("today") + dur("7 days")'
    assert_includes alert, '证据状态 = "待校准"'
    assert_match(/LIMIT\s+8\z/i, alert)

    scores = queries.fetch("本周得分链")
    %w[思想得分 正确性得分 代码得分 复杂度得分 总分].each { |term| assert_includes scores, term }
    assert_includes scores, 'date(choice(最后训练, 最后训练, "2999-12-31")) >= date("today") - dur("7 days")'
    assert_match(/LIMIT\s+50\z/i, scores)

    stages = queries.fetch("训练阶段分布")
    assert_includes stages, "GROUP BY 状态"
    assert_includes stages, "length(rows)"
  end

  def test_body_contains_only_the_seven_empty_headings_in_matching_order
    headings = @body.scan(/^## (.+)$/).flatten
    assert_equal COLUMN_NAMES, headings

    sections = @body.split(/^## .+\n/, -1).drop(1)
    assert_equal COLUMN_NAMES.length, sections.length
    sections.each { |content| assert_empty content.strip }
  end

  private

  def parse_json(path)
    JSON.parse(path.read(encoding: "UTF-8"))
  end

  def parse_frontmatter(path)
    raw = path.read(encoding: "UTF-8")
    match = raw.match(/\A---\r?\n(.*?)\r?\n---\r?\n?(.*)\z/m)
    refute_nil match, "#{path} must have closed leading YAML frontmatter"
    frontmatter = YAML.safe_load(match[1], aliases: false)
    assert_instance_of Hash, frontmatter
    [frontmatter, match[2]]
  end

  def assert_order(text, *needles)
    offsets = needles.map do |needle|
      offset = text.index(needle)
      refute_nil offset, "missing #{needle.inspect} from query"
      offset
    end
    assert_equal offsets.sort, offsets, "expected priority clauses in approved order"
  end

  def assert_balanced_dql(query)
    quote = nil
    escaped = false
    depth = 0
    query.each_char do |char|
      if quote
        if escaped
          escaped = false
        elsif char == "\\"
          escaped = true
        elsif char == quote
          quote = nil
        end
      elsif char == '"' || char == "'"
        quote = char
      elsif char == "("
        depth += 1
      elsif char == ")"
        depth -= 1
        assert_operator depth, :>=, 0, "unexpected closing parenthesis"
      end
    end
    assert_nil quote, "unterminated DQL string"
    assert_equal 0, depth, "unbalanced DQL parentheses"
    assert_match(/\ATABLE\s+/i, query)
    assert_match(/\sFROM\s+"11_数据结构\//i, query)
  end
end
