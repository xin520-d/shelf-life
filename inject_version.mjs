// postbuild: 给 dist/index.html 的资源引用追加 ?v=<时间戳>，使每次构建都自动破缓存。
// 配合后端 SPA 回退的 Cache-Control: no-cache，刷新/新开页面即可拿到最新前端，无需手工改版本号。
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const __dirname = dirname(fileURLToPath(import.meta.url))
const htmlPath = join(__dirname, 'dist', 'index.html')

const version = String(Date.now())
let html = readFileSync(htmlPath, 'utf-8')

// 仅给尚未带 ?v 的本地资源（./assets/...）追加版本号
html = html.replace(/(src|href)="(\.\/assets\/[^"]+?)(\?v=[^"]*)?"/g, (_m, attr, url) => {
  return `${attr}="${url}?v=${version}"`
})

writeFileSync(htmlPath, html, 'utf-8')
console.log(`[inject_version] stamped dist/index.html assets with ?v=${version}`)
