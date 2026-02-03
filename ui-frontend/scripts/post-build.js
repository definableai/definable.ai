/**
 * Post-build script to prepare Web Component for CDN distribution
 * 
 * This script:
 * 1. Copies build files to the Python static directory
 * 2. Creates a simple HTML loader for the web component
 * 3. Generates version info
 */

import { readFileSync, writeFileSync, copyFileSync, mkdirSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const distDir = resolve(__dirname, '../dist');
const staticDir = resolve(__dirname, '../../definable/definable/ui/static');

// Ensure static directory exists
if (!existsSync(staticDir)) {
  mkdirSync(staticDir, { recursive: true });
}

console.log('📦 Post-build: Preparing Web Component for CDN...');

// Copy all dist files to static directory
console.log('📋 Copying build files to static directory...');
const fs = await import('fs');
const files = fs.readdirSync(distDir);

files.forEach(file => {
  const src = resolve(distDir, file);
  const dest = resolve(staticDir, file);
  
  if (fs.statSync(src).isFile()) {
    copyFileSync(src, dest);
    console.log(`  ✓ Copied ${file}`);
  }
});

// Create a simple HTML loader
const loaderHtml = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Definable Chat</title>
  <style>
    body {
      margin: 0;
      padding: 0;
      height: 100vh;
      overflow: hidden;
    }
    definable-chat {
      display: block;
      width: 100%;
      height: 100%;
    }
  </style>
</head>
<body>
  <!-- Definable Chat Web Component -->
  <definable-chat></definable-chat>
  
  <!-- Load the web component -->
  <script type="module" src="./definable-chat.es.js"></script>
</body>
</html>
`;

writeFileSync(resolve(staticDir, 'index.html'), loaderHtml);
console.log('  ✓ Created index.html loader');

// Create version info
const packageJson = JSON.parse(readFileSync(resolve(__dirname, '../package.json'), 'utf-8'));
const versionInfo = {
  version: packageJson.version,
  buildTime: new Date().toISOString(),
  component: 'definable-chat'
};

writeFileSync(
  resolve(staticDir, 'version.json'),
  JSON.stringify(versionInfo, null, 2)
);
console.log('  ✓ Created version.json');

// Create CDN usage instructions
const cdnReadme = `# Definable Chat Web Component

## CDN Usage

### Via Script Tag

\`\`\`html
<!DOCTYPE html>
<html>
<head>
  <style>
    body { margin: 0; padding: 0; height: 100vh; }
    definable-chat { display: block; width: 100%; height: 100%; }
  </style>
</head>
<body>
  <!-- Web Component -->
  <definable-chat 
    title="My AI Assistant"
    theme="dark"
    ws-url="ws://localhost:8000/ws"
  ></definable-chat>
  
  <!-- Load from CDN -->
  <script type="module" src="https://cdn.example.com/definable-chat.es.js"></script>
</body>
</html>
\`\`\`

## Attributes

- **title**: Chat window title (default: "Definable Chat")
- **theme**: Theme mode - "light" or "dark" (default: "light")
- **ws-url**: WebSocket URL (default: "ws://[current-host]/ws")

## JavaScript API

\`\`\`javascript
const chat = document.querySelector('definable-chat');

// Update attributes dynamically
chat.setAttribute('title', 'New Title');
chat.setAttribute('theme', 'dark');
\`\`\`

## Version

${versionInfo.version}
Built: ${versionInfo.buildTime}
`;

writeFileSync(resolve(staticDir, 'README.md'), cdnReadme);
console.log('  ✓ Created README.md');

console.log('\n✅ Post-build complete!');
console.log(`\n📍 Files ready for CDN at: ${staticDir}`);
console.log('\nWeb Component can be used as:');
console.log('  <definable-chat title="My Chat" theme="dark"></definable-chat>');
