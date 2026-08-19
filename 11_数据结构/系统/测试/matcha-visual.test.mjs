import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { homedir } from 'node:os';
import { readFileSync, existsSync } from 'node:fs';

const root = new URL('../../../', import.meta.url);
const require = createRequire(import.meta.url);
const postcss = require(`${homedir()}/Documents/408/node_modules/postcss`);
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

function splitFrontmatter(source) {
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  assert.ok(match, 'page must retain YAML frontmatter');
  return { yaml: match[1], body: source.slice(match[0].length) };
}

test('snippet exists, parses, and every selector branch is project-scoped', () => {
  assert.ok(existsSync(cssPath), '408-training.css is missing');
  assert.ok(rules().length > 0, 'snippet must contain CSS rules');
  assert.deepEqual(selectors().filter((selector) => !scopes.test(selector)), []);
  ast.walkDecls((decl) => assert.equal(Boolean(decl.important), false, `${decl.prop} must not use !important`));
});

test('Matcha tokens, max width, dark mode, focus and responsive contracts exist', () => {
  const tokens = declarations('.ds-408-dashboard');
  for (const [property, value] of Object.entries({
    '--ds408-matcha-900': '#30483a', '--ds408-matcha-700': '#58725f', '--ds408-matcha-500': '#7f9d82',
    '--ds408-matcha-200': '#cbd8c7', '--ds408-paper': '#f6f2e7', '--ds408-ink': '#26342c',
    '--h1-color': 'var(--ds408-matcha-900)', '--h2-color': 'var(--ds408-matcha-700)',
    '--link-color': 'var(--ds408-matcha-700)', '--interactive-accent': 'var(--ds408-matcha-700)'
  })) assert.equal(tokens.get(property), value, property);
  for (const scope of ['dashboard', 'problem', 'pattern']) {
    assert.equal(declarations(`.ds-408-${scope} .markdown-preview-sizer`).get('max-width'), '1120px');
    assert.equal(declarations(`.theme-dark .ds-408-${scope}`).get('--ds408-paper'), '#202822');
  }
  assert.ok(hasSelector(':focus-visible'));
  assert.ok(hasAtRule('media', 'max-width: 700px'));
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
