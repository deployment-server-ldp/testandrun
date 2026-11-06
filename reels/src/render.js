const { chromium } = require('playwright');
const { spawn } = require('child_process');
const path = require('path');
(async () => {
  const mode = process.argv[2]; // "stills" or "video"
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
  await p.goto('file://' + path.resolve('reel.html'));
  await p.evaluate(() => document.fonts.ready);
  if (mode === 'stills') {
    for (const t of [1.5, 4.0, 5.5, 7.5, 11.5, 15.2, 19]) {
      await p.evaluate(t => render(t), t);
      await p.screenshot({ path: `still_${t}.png` });
    }
  } else {
    const FPS = 30, DUR = 20;
    const ff = spawn('ffmpeg', ['-y','-f','image2pipe','-framerate',String(FPS),'-i','-',
      '-f','lavfi','-i','anullsrc=r=44100:cl=stereo','-shortest',
      '-c:v','libx264','-pix_fmt','yuv420p','-crf','18','-preset','medium','-r',String(FPS),
      '-c:a','aac','-movflags','+faststart','out.mp4'], { stdio: ['pipe','ignore','inherit'] });
    for (let i = 0; i < FPS*DUR; i++) {
      await p.evaluate(t => render(t), i/FPS);
      const buf = await p.screenshot({ type: 'jpeg', quality: 92 });
      if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    }
    ff.stdin.end();
    await new Promise(r => ff.on('close', r));
  }
  await b.close();
})();
