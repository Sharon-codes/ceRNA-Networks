const { chromium } = require('C:/Users/Samsunh/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const path = require('path');
const fs = require('fs');
const { pathToFileURL } = require('url');
(async()=>{
 const dir=path.resolve('research_rebuild/outputs/20260907_biosystems_evidence_v1');
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1400,height:1000},deviceScaleFactor:1});
 await page.goto(pathToFileURL(path.join(dir,'FINAL_REPORT.html')).href,{waitUntil:'load'});
 await page.evaluate(()=>document.fonts.ready);
 const check=await page.evaluate(()=>({images:[...document.images].map(i=>({loaded:i.complete,width:i.naturalWidth,srcKind:i.src.slice(0,25)})),horizontalOverflow:document.documentElement.scrollWidth>innerWidth,headings:[...document.querySelectorAll('h2')].map(h=>h.textContent)}));
 if(check.images.some(i=>!i.loaded||i.width===0))throw Error('Image loading failure');
 await page.pdf({path:path.join(dir,'FINAL_REPORT.pdf'),preferCSSPageSize:true,printBackground:true});
 await page.screenshot({path:path.join(dir,'figures/report_first_screen.png')});
 for(let n=0;n<check.images.length;n++)await page.locator('img').nth(n).screenshot({path:path.join(dir,'figures',`report_embedded_figure${n+1}.png`)});
 fs.writeFileSync(path.join(dir,'logs/render_check.json'),JSON.stringify(check,null,2));
 await browser.close();console.log('Rendered PDF; images:',check.images.length,'horizontal overflow:',check.horizontalOverflow);
})();
