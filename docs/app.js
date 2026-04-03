const routes = {
  'index': 'pages/index.md',
  'architecture': 'pages/architecture.md',
  'data_pipeline': 'pages/data_pipeline.md',
  'training_and_generation': 'pages/training_and_generation.md',
  'configuration': 'pages/configuration.md'
};

const contentDiv = document.getElementById('content');
const navLinks = document.querySelectorAll('nav a');

// Setup marked.js renderer
const renderer = new marked.Renderer();
renderer.code = function(codeOrToken, language, isEscaped) {
  let code = codeOrToken;
  let lang = language;

  // 最新の marked.js では引数に Token オブジェクトが一つだけ渡されます
  if (typeof codeOrToken === 'object' && codeOrToken !== null) {
      code = codeOrToken.text;
      lang = codeOrToken.lang;
  }

  // HTMLタグ等がそのまま解釈されないようにエスケープ処理
  const escapedCode = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');

  if (lang === 'mermaid') {
    return `<div class="mermaid">${code}</div>`;
  }
  return `<pre><code class="language-${lang || ''}">${escapedCode}</code></pre>`;
};
marked.setOptions({ renderer });

// Custom HTML entities decoding for safety (Mermaid sometimes needs it)
const decodeHTML = (html) => {
  let txt = document.createElement("textarea");
  txt.innerHTML = html;
  return txt.value;
}

async function loadPage(page) {
  const filename = routes[page];
  if (!filename) return;

  // Set active nav link
  navLinks.forEach(link => {
    if (link.getAttribute('data-page') === page) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });

  contentDiv.innerHTML = '<div class="loader"></div>';

  try {
    const response = await fetch(filename);
    if (!response.ok) {
        throw new Error(`Could not fetch ${filename}`);
    }
    const markdown = await response.text();
    
    // Convert Markdown to HTML
    let html = marked.parse(markdown);
    
    contentDiv.innerHTML = html;

    // Apply syntax highlighting
    document.querySelectorAll('pre code').forEach((block) => {
      hljs.highlightElement(block);
    });

    // Render Mermaid diagrams
    mermaid.init(undefined, document.querySelectorAll('.mermaid'));

    // Render Math (KaTeX)
    if (window.renderMathInElement) {
      renderMathInElement(contentDiv, {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '$', right: '$', display: false},
          {left: '\\(', right: '\\)', display: false},
          {left: '\\[', right: '\\]', display: true}
        ],
        throwOnError: false
      });
    }

    // Intercept local links inside the markdown content
    document.querySelectorAll('#content a').forEach(a => {
      const href = a.getAttribute('href');
      if (href && href.endsWith('.md')) {
        a.addEventListener('click', (e) => {
          e.preventDefault();
          const targetPage = href.replace('.md', '');
          // Fix for paths like './architecture.md'
          const cleanPage = targetPage.replace(/^\.\//, '');
          if (routes[cleanPage]) {
            window.location.hash = cleanPage;
          }
        });
      }
    });

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });

  } catch (error) {
    contentDiv.innerHTML = `
        <div style="background: rgba(255, 99, 132, 0.1); border: 1px solid rgba(255, 99, 132, 0.3); padding: 1rem; border-radius: 8px; color: #ff6384;">
            <h3>Error loading content</h3>
            <p>${error.message}</p>
            <p style="margin-top: 1rem; font-size: 0.9em;">Note: Since this uses fetch() to load markdown files, you cannot just double-click the HTML file in some browsers due to CORS restrictions.</p>
            <p style="font-weight: bold; margin-top: 0.5rem;">Please start a local server, e.g.: <code>python -m http.server 8000</code> and go to <code>http://localhost:8000/docs/</code></p>
        </div>`;
  }
}

// Handle routing via Hash
function handleRoute() {
  let hash = window.location.hash.substring(1);
  if (!hash || !routes[hash]) {
    hash = 'index';
  }
  loadPage(hash);
}

// Initialize Mermaid
mermaid.initialize({ 
    startOnLoad: false,
    theme: 'default'
});

// Event Listeners
window.addEventListener('hashchange', handleRoute);

// Bootstrap
handleRoute();
