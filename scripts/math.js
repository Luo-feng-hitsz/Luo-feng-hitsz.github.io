'use strict';

const { escapeHTML } = require('hexo-util');

hexo.extend.filter.register('marked:extensions', (extensions) => {
  extensions.push(
    {
      name: 'blockMath',
      level: 'block',
      start(src) {
        return src.indexOf('$$');
      },
      tokenizer(src) {
        const match = /^\s{0,3}\$\$[ \t]*\n([\s\S]+?)\n\$\$[ \t]*(?:\n|$)/.exec(src);
        if (!match) return undefined;

        return {
          type: 'blockMath',
          raw: match[0],
          math: match[1]
        };
      },
      renderer(token) {
        return `<div class="math-display">$$${escapeHTML(token.math)}$$</div>\n`;
      }
    },
    {
      name: 'inlineMath',
      level: 'inline',
      start(src) {
        return src.indexOf('$');
      },
      tokenizer(src) {
        const match = /^\$(?!\$)((?:\\\$|[^$\n])+?)\$(?!\$)/.exec(src);
        if (!match) return undefined;

        return {
          type: 'inlineMath',
          raw: match[0],
          math: match[1]
        };
      },
      renderer(token) {
        return `<span class="math-inline">$${escapeHTML(token.math)}$</span>`;
      }
    }
  );
});
