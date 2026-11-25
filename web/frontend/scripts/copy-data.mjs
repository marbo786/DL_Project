import { copyFile, mkdir, readFile } from 'node:fs/promises';

const output = new URL('../public/data/', import.meta.url);
await mkdir(output, { recursive: true });

for (const name of ['results.json', 'sample-results.json']) {
  const source = new URL('../data/' + name, import.meta.url);
  JSON.parse(await readFile(source, 'utf8'));
  await copyFile(source, new URL(name, output));
}
