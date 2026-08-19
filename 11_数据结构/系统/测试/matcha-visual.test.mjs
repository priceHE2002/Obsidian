import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { readFileSync, existsSync } from 'node:fs';

const require = createRequire(import.meta.url);
const postcss = require('postcss');
const cssPath = new URL('../../../.obsidian/snippets/408-training.css', import.meta.url);
const css = existsSync(cssPath) ? readFileSync(cssPath, 'utf8') : '';
const ast = postcss.parse(css);
const scopes = /^(?:\.theme-dark\s+)?\.ds-408-(?:dashboard|problem|pattern)(?=$|[\s.:#\[])/;
const branches = (selector) => postcss.list.comma(selector);
const rules = () => { const found = []; ast.walkRules((rule) => found.push(rule)); return found; };
const selectors = () => rules().flatMap((rule) => branches(rule.selector));
const hasSelector = (fragment) => selectors().some((selector) => selector.includes(fragment));
const hasAtRule = (name, fragment) => { let hit = false; ast.walkAtRules(name, (node) => { if (node.params.includes(fragment)) hit = true; }); return hit; };
const declarations = (selector) => { const map = new Map(); rules().filter((rule) => rule.parent.type === 'root' && branches(rule.selector).includes(selector)).forEach((rule) => rule.walkDecls((decl) => map.set(decl.prop, decl.value))); return map; };

function cssContractViolations(source) {
  const parsed = postcss.parse(source);
  const violations = [];
  const allowedMedia = new Set(['(max-width: 700px)', 'print']);
  parsed.walkAtRules((atRule) => {
    const allowed = atRule.name === 'media' && allowedMedia.has(atRule.params.trim()) && atRule.parent.type === 'root';
    if (!allowed) violations.push(`unsupported at-rule: @${atRule.name} ${atRule.params}`.trim());
  });
  parsed.walkRules((rule) => branches(rule.selector).forEach((selector) => {
    if (!scopes.test(selector)) violations.push(`unscoped selector: ${selector}`);
  }));
  return violations;
}

function onlyMedia(params) {
  const matches = [];
  ast.walkAtRules('media', (atRule) => { if (atRule.params.trim() === params) matches.push(atRule); });
  assert.equal(matches.length, 1, `expected exactly one @media ${params}`);
  return matches[0];
}

function mediaHas(media, selectorFragment, property, value) {
  let found = false;
  media.walkRules((rule) => {
    if (!rule.selector.includes(selectorFragment)) return;
    rule.walkDecls(property, (decl) => { if (decl.value === value) found = true; });
  });
  return found;
}

function exactRule(selector, container = ast) {
  const matches = [];
  container.walkRules((rule) => { if (branches(rule.selector).includes(selector)) matches.push(rule); });
  assert.equal(matches.length, 1, `expected one rule branch for ${selector}`);
  return matches[0];
}

function directDeclarations(rule) {
  return new Map(rule.nodes.filter((node) => node.type === 'decl').map((decl) => [decl.prop, decl.value]));
}

function rgb(hex) {
  assert.match(hex, /^#[0-9a-f]{6}$/i);
  return [1, 3, 5].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16));
}

function mix(foreground, background, foregroundWeight) {
  return rgb(foreground).map((channel, index) => channel * foregroundWeight + rgb(background)[index] * (1 - foregroundWeight));
}

function luminance(color) {
  const channels = Array.isArray(color) ? color : rgb(color);
  return channels.map((value) => value / 255).map((value) => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4)
    .reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0);
}

function contrast(a, b) {
  const [lighter, darker] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (lighter + 0.05) / (darker + 0.05);
}

function focusSources() {
  const sources = [];
  for (const folder of ['状态控件', '题目导航', '复习队列', '题库表', '统计卡']) {
    const sourceAst = postcss.parse(readFileSync(new URL(`../${folder}/view.css`, import.meta.url), 'utf8'));
    sourceAst.walkRules((rule) => branches(rule.selector).filter((selector) => selector.includes(':focus-visible')).forEach((selector) => sources.push({ folder, selector })));
  }
  return sources;
}

function classAndPseudoSpecificity(selector) {
  const classes = selector.match(/\.[a-z0-9_-]+/gi) || [];
  const attributes = selector.match(/\[[^\]]+\]/g) || [];
  const pseudos = (selector.match(/:(?!:)[a-z0-9_-]+/gi) || []).filter((name) => name !== ':is');
  return classes.length + attributes.length + pseudos.length;
}

function splitFrontmatter(source) {
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  assert.ok(match, 'page must retain YAML frontmatter');
  return { yaml: match[1], body: source.slice(match[0].length) };
}

test('snippet exists, parses, and every selector branch is project-scoped', () => {
  assert.ok(existsSync(cssPath), '408-training.css is missing');
  assert.ok(rules().length > 0, 'snippet must contain CSS rules');
  assert.deepEqual(cssContractViolations(css), []);
  ast.walkDecls((decl) => assert.equal(Boolean(decl.important), false, `${decl.prop} must not use !important`));
});

test('root-level global at-rules are rejected by the CSS contract', () => {
  for (const probe of [
    '@font-face { font-family: Leak; src: url(leak.woff2); }',
    '@import url("leak.css");',
    '@property --leak { syntax: "<color>"; inherits: true; initial-value: red; }',
    '@page { margin: 1cm; }'
  ]) assert.match(cssContractViolations(probe).join('\n'), /unsupported at-rule/);
});

test('Matcha tokens, max width, dark mode, focus and responsive contracts exist', () => {
  const tokens = declarations('.ds-408-dashboard');
  for (const [property, value] of Object.entries({
    '--ds408-matcha-900': '#30483a', '--ds408-matcha-700': '#58725f', '--ds408-matcha-500': '#7f9d82',
    '--ds408-matcha-200': '#cbd8c7', '--ds408-paper': '#f6f2e7', '--ds408-ink': '#26342c',
    '--h1-color': 'var(--ds408-strong-text)', '--h2-color': 'var(--ds408-matcha-700)',
    '--link-color': 'var(--ds408-matcha-700)', '--interactive-accent': 'var(--ds408-matcha-700)'
  })) assert.equal(tokens.get(property), value, property);
  for (const scope of ['dashboard', 'problem', 'pattern']) {
    assert.equal(declarations(`.ds-408-${scope} .markdown-preview-sizer`).get('max-width'), '1120px');
    assert.equal(declarations(`.theme-dark .ds-408-${scope}`).get('--ds408-paper'), '#202822');
  }
  assert.ok(hasSelector(':focus-visible'));
  assert.ok(hasAtRule('media', 'max-width: 700px'));
});

test('light and dark semantic foreground tokens meet WCAG contrast', () => {
  const light = declarations('.ds-408-dashboard');
  const dark = declarations('.theme-dark .ds-408-dashboard');
  assert.equal(light.get('--ds408-strong-text'), '#30483a');
  assert.equal(dark.get('--ds408-strong-text'), '#dce8dc');
  assert.equal(light.get('--ds408-warning-text'), '#7a3f3a');
  assert.equal(dark.get('--ds408-warning-text'), '#f0aaa2');
  assert.equal(light.get('--ds408-focus'), '#58725f');
  assert.equal(dark.get('--ds408-focus'), '#aac3ad');
  for (const background of ['#f6f2e7', '#ffffff', mix('#cbd8c7', '#ffffff', 0.72), mix('#cbd8c7', '#ffffff', 0.62)])
    assert.ok(contrast(light.get('--ds408-strong-text'), background) >= 4.5, 'light strong text contrast');
  for (const background of ['#202822', mix('#435548', '#202822', 0.72), mix('#435548', '#202822', 0.62)])
    assert.ok(contrast(dark.get('--ds408-strong-text'), background) >= 4.5, 'dark strong text contrast');
  assert.ok(contrast(light.get('--ds408-warning-text'), mix('#9f5f58', '#ffffff', 0.14)) >= 4.5, 'light warning contrast');
  assert.ok(contrast(dark.get('--ds408-warning-text'), mix('#9f5f58', '#202822', 0.14)) >= 4.5, 'dark warning contrast');
  assert.ok(contrast(light.get('--ds408-focus'), '#f6f2e7') >= 3, 'light focus contrast');
  assert.ok(contrast(dark.get('--ds408-focus'), '#202822') >= 3, 'dark focus contrast');
});

test('real component overrides have three-class specificity and semantic declarations', () => {
  const contracts = [
    ['.ds408-kpi .ds408-kpi__card', 'background', 'var(--ds408-surface)'],
    ['.ds408-review-queue .ds408-review-queue__card', 'background', 'var(--ds408-surface)'],
    ['.ds408-control .ds408-control__badge', 'color', 'var(--ds408-strong-text)'],
    ['.ds408-problem-nav .ds408-problem-nav__link', 'color', 'var(--ds408-strong-text)'],
    ['.ds408-bank .ds408-bank__table-wrap', 'overflow-x', 'auto'],
    ['.ds408-bank .ds408-bank__table', 'background', 'var(--ds408-surface)'],
    ['.ds408-bank .ds408-bank__table th', 'color', 'var(--ds408-strong-text)']
  ];
  for (const scope of ['dashboard', 'problem', 'pattern']) for (const [suffix, property, value] of contracts) {
    const selector = `.ds-408-${scope} ${suffix}`;
    assert.equal((selector.match(/\.[a-z0-9_-]+/gi) || []).length, 3, `${selector} specificity`);
    assert.equal(directDeclarations(exactRule(selector)).get(property), value, `${selector} ${property}`);
  }
});

test('focus and callout borders use valid direct semantic tokens', () => {
  const focus = exactRule('.ds-408-dashboard :is(button, input, select):focus-visible');
  assert.equal(directDeclarations(focus).get('outline'), '3px solid var(--ds408-focus)');
  const callout = directDeclarations(exactRule('.ds-408-dashboard .callout'));
  assert.equal(callout.get('border-left-color'), 'rgb(var(--callout-color))');
  assert.equal(callout.get('border-left-width'), '4px');
  assert.equal(callout.has('border-left'), false);
  ast.walkDecls('color', (decl) => assert.notEqual(decl.value, 'var(--ds408-matcha-900)'));
  const warning = directDeclarations(exactRule('.ds-408-dashboard :is(.ds408-control__badge.is-warning, .ds408-evidence-badge[data-evidence="待校准"])'));
  assert.equal(warning.get('color'), 'var(--ds408-warning-text)');
});

test('every real component focus rule has a strictly stronger semantic snippet override', () => {
  const sources = focusSources();
  assert.deepEqual(sources.map(({ folder, selector }) => `${folder}:${selector}`).sort(), [
    '复习队列:.ds408-review-queue .ds408-review-queue__link:focus-visible',
    '状态控件:.ds408-control :is(button, input, select):focus-visible',
    '统计卡:.ds408-kpi .ds408-kpi__card:focus-visible',
    '题库表:.ds408-bank .ds408-bank__cancel:focus-visible',
    '题库表:.ds408-bank .ds408-bank__clear:focus-visible',
    '题库表:.ds408-bank .ds408-bank__filter select:focus-visible',
    '题库表:.ds408-bank .ds408-bank__link:focus-visible',
    '题库表:.ds408-bank .ds408-bank__save:focus-visible',
    '题库表:.ds408-bank .ds408-bank__sort:focus-visible',
    '题库表:.ds408-bank .ds408-bank__status:focus-visible',
    '题目导航:.ds408-problem-nav .ds408-problem-nav__link:focus-visible'
  ].sort());
  for (const scope of ['dashboard', 'problem', 'pattern']) for (const { selector: source } of sources) {
    const override = `.ds-408-${scope} ${source}`;
    assert.ok(classAndPseudoSpecificity(override) > classAndPseudoSpecificity(source), `${override} must outrank late component CSS`);
    const declarations = directDeclarations(exactRule(override));
    assert.equal(declarations.get('outline'), '3px solid var(--ds408-focus)', override);
    assert.equal(declarations.get('outline-offset'), '3px', override);
  }
});

test('mobile and print declarations stay inside their approved media blocks', () => {
  const mobile = onlyMedia('(max-width: 700px)');
  assert.ok(mediaHas(mobile, '.ds408-kpi', 'grid-template-columns', 'minmax(0, 1fr)'));
  assert.ok(mediaHas(mobile, '.table-wrapper', 'overflow-x', 'auto'));
  const wrapper = directDeclarations(exactRule('.ds-408-dashboard :is(.table-wrapper, .block-language-dataview, .ds408-bank__table-wrap)', mobile));
  assert.equal(wrapper.get('width'), '100%');
  assert.equal(wrapper.get('max-width'), '100%');
  assert.equal(wrapper.get('box-sizing'), 'border-box');
  assert.equal(wrapper.get('min-width'), '0');
  const print = onlyMedia('print');
  assert.ok(mediaHas(print, '.ds408-control', 'display', 'none'));
  assert.ok(mediaHas(print, '.ds408-kpi__card', 'box-shadow', 'none'));
  for (const scope of ['dashboard', 'problem', 'pattern']) {
    assert.equal(directDeclarations(exactRule(`.ds-408-${scope}`, print)).get('--ds408-shadow'), 'none');
    assert.equal(directDeclarations(exactRule(`.ds-408-${scope} .ds408-kpi .ds408-kpi__card`)).get('box-shadow'), 'var(--ds408-shadow)');
    assert.equal(directDeclarations(exactRule(`.ds-408-${scope} .ds408-review-queue .ds408-review-queue__card`)).get('box-shadow'), 'var(--ds408-shadow)');
  }
});

test('snippet covers real 408 cards, badges, content, controls and internal table scrolling', () => {
  for (const fragment of [
    '.ds408-kpi__card', '.ds408-review-queue__card', '.ds408-control__badge', '.ds408-control__badge.is-warning',
    '.callout', '.table-view-table', '.ds408-bank__table-wrap', ' pre', ' code', '.metadata-container',
    '.metadata-property', '.ds408-problem-nav__link', ' button', ' input', ' select'
  ]) assert.ok(hasSelector(fragment), `missing real selector coverage: ${fragment}`);
  const scrollRules = rules().filter((rule) => [...branches(rule.selector)].some((selector) => /table-view-table|table-wrapper|table-wrap|block-language-dataview/.test(selector)));
  assert.ok(scrollRules.some((rule) => rule.nodes.some((node) => node.type === 'decl' && node.prop === 'overflow-x' && node.value === 'auto')));
  for (const rule of rules()) {
    if (!rule.nodes.some((node) => node.type === 'decl' && node.prop === 'overflow-x')) continue;
    assert.ok(branches(rule.selector).every((selector) => /table-view-table|table-wrapper|table-wrap|block-language-dataview|\bpre\b/.test(selector)), `page-level overflow on ${rule.selector}`);
  }
});

test('appearance keeps font size and enables the snippet exactly once', () => {
  const appearance = JSON.parse(readFileSync(new URL('../../../.obsidian/appearance.json', import.meta.url), 'utf8'));
  assert.equal(appearance.baseFontSize, 14);
  assert.equal(appearance.enabledCssSnippets.filter((name) => name === '408-training').length, 1);
});

test('six entry pages have exactly one dashboard class and unchanged bodies', () => {
  const expected = {
    '总览.md': '196cf89c8b4b4c2216c7a332750520dab3636271da8524ea89de38bfd926c6e7',
    '408作战台.md': 'ce03aaaf814dc6789831304621e337691c58d41856a0a08d14f9a4d9637908ea',
    '真题/真题冲刺.md': 'e06ce69a30dc4b540a83cdef6aa81e3bb838324e7dedc02678d66e38c7ebfb37',
    '错题回炉.md': '69f4f9fb73680ce23167435de119dbde707cb7085f4bb44b1046af28e34a5997',
    '训练数据校准.md': '49534b663d14110a8937785f684b1dff70867b80ab9063aabb2c7e2d5fced5ab',
    '使用说明.md': '84615839db4106351c8c602ac316f5bacd994243b92c42f6fcea1f7c8a366a55'
  };
  for (const [relative, digest] of Object.entries(expected)) {
    const page = splitFrontmatter(readFileSync(new URL(`../../${relative}`, import.meta.url), 'utf8'));
    assert.equal((page.yaml.match(/^\s*- ds-408-dashboard\s*$/gm) || []).length, 1, relative);
    assert.equal(createHash('sha256').update(page.body).digest('hex'), digest, `${relative} body changed`);
  }
});
