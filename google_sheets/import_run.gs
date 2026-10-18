/**
 * This file is a Google Apps Script that imports one run's CSV files into
 * the open Google Sheet, one tab per file: Summary, Room Order, Corner
 * Motion (and a tab for any other CSV in the run folder).
 *
 * Setup, once per spreadsheet:
 *   1. In the Google Sheet: Extensions > Apps Script.
 *   2. Replace the editor's contents with this whole file and save.
 *   3. Reload the sheet. A "Tactile Maps" menu appears.
 *
 * Use: Tactile Maps > Import run CSVs..., then pick the run folder
 * (data/<date_time>__<map>__<video>/) or just its CSV files. Importing
 * again replaces those tabs; other tabs are left alone. The first time,
 * Google asks you to authorize the script for this spreadsheet.
 *
 * The CSVs come from data_export.py: plain tables, header row first, times
 * in seconds. Numbers are written as numbers; everything else as plain text,
 * so a cell like "(8.5, 11)" isn't reinterpreted by Sheets.
 */

// CSV file name -> tab name, in tab order. Other CSVs follow, named from
// their file name ("finger_speed.csv" -> "Finger Speed").
var TABS = [
  ['summary.csv', 'Summary'],
  ['room_order.csv', 'Room Order'],
  ['corner_motion.csv', 'Corner Motion'],
];

var NUMBER = /^-?\d+(\.\d+)?$/;

function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('Tactile Maps')
    .addItem('Import run CSVs...', 'showImportDialog')
    .addToUi();
}

function showImportDialog() {
  var html = HtmlService.createHtmlOutput(DIALOG_HTML).setWidth(420).setHeight(240);
  SpreadsheetApp.getUi().showModalDialog(html, 'Import run CSVs');
}

/**
 * Called from the dialog. files: [{name: "summary.csv", text: "..."}];
 * runName: the picked folder's name, or "" when single files were picked.
 * Returns a one-line report for the dialog.
 */
function importCsvs(files, runName) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var csvs = files.filter(function (f) { return /\.csv$/i.test(f.name); });
  if (!csvs.length) throw new Error('No .csv files were picked.');
  csvs.sort(function (a, b) { return tabOrder(a.name) - tabOrder(b.name); });

  var blank = ss.getSheets().filter(isBlankSheet);
  var report = [];
  csvs.forEach(function (f, i) {
    var name = tabName(f.name);
    var sheet = ss.getSheetByName(name) || ss.insertSheet(name);
    writeTable(sheet, Utilities.parseCsv(f.text));
    ss.setActiveSheet(sheet);
    ss.moveActiveSheet(i + 1);
    report.push(name + ' (' + Math.max(sheet.getLastRow() - 1, 0) + ' rows)');
  });
  // Drop the empty "Sheet1" a new spreadsheet starts with.
  blank.forEach(function (s) {
    if (ss.getSheets().length > 1 && isBlankSheet(s)) ss.deleteSheet(s);
  });
  if (runName && /^Untitled/.test(ss.getName())) ss.rename(runName);
  ss.setActiveSheet(ss.getSheets()[0]);
  return 'Imported ' + report.join(', ') + '.';
}

function writeTable(sheet, rows) {
  sheet.clear();
  sheet.setFrozenRows(0);
  while (rows.length && rows[rows.length - 1].join('') === '') rows.pop();   // trailing blank lines
  if (!rows.length) return;
  var width = Math.max.apply(null, rows.map(function (r) { return r.length; }));
  var values = rows.map(function (r, i) {
    var row = r.slice();
    while (row.length < width) row.push('');
    return i === 0 ? row : row.map(cellValue);
  });
  var range = sheet.getRange(1, 1, values.length, width);
  // Text columns (any non-number cell below the header) are set to plain
  // text first, so Sheets keeps "(8.5, 11)" or a room name exactly as is.
  for (var c = 0; c < width; c++) {
    var isText = values.slice(1).some(function (row) { return typeof row[c] === 'string' && row[c] !== ''; });
    if (isText || values.length === 1) sheet.getRange(1, c + 1, values.length, 1).setNumberFormat('@');
  }
  range.setValues(values);
  sheet.getRange(1, 1, 1, width).setFontWeight('bold');
  sheet.setFrozenRows(1);
  sheet.autoResizeColumns(1, width);
}

function cellValue(cell) {
  var s = String(cell).trim();
  return NUMBER.test(s) ? Number(s) : s;
}

function tabName(fileName) {
  var base = fileName.split('/').pop();
  for (var i = 0; i < TABS.length; i++) {
    if (TABS[i][0] === base.toLowerCase()) return TABS[i][1];
  }
  return base.replace(/\.csv$/i, '').split(/[_\s-]+/).map(function (w) {
    return w.charAt(0).toUpperCase() + w.slice(1);
  }).join(' ').slice(0, 100);
}

function tabOrder(fileName) {
  var base = fileName.split('/').pop().toLowerCase();
  for (var i = 0; i < TABS.length; i++) {
    if (TABS[i][0] === base) return i;
  }
  return TABS.length;
}

function isBlankSheet(sheet) {
  return sheet.getLastRow() === 0 && sheet.getLastColumn() === 0 &&
    !TABS.some(function (t) { return t[1] === sheet.getName(); });
}

var DIALOG_HTML = [
  '<!doctype html><html><head><base target="_top"><style>',
  'body{font:14px Arial,sans-serif;margin:0 4px}',
  'label{display:block;margin:10px 0 4px;font-weight:bold}',
  '#status{margin-top:14px;min-height:2.5em}',
  'button{margin-top:12px;padding:6px 16px}',
  '</style></head><body>',
  '<label for="folder">Run folder</label>',
  '<input type="file" id="folder" webkitdirectory>',
  '<label for="files">or CSV files</label>',
  '<input type="file" id="files" multiple accept=".csv,text/csv">',
  '<br><button id="go">Import</button>',
  '<div id="status"></div>',
  '<script>',
  'var status = document.getElementById("status");',
  'document.getElementById("go").onclick = function () {',
  '  var folder = document.getElementById("folder").files;',
  '  var picked = Array.prototype.slice.call(folder.length ? folder : document.getElementById("files").files)',
  '    .filter(function (f) { return /\\.csv$/i.test(f.name); });',
  '  if (!picked.length) { status.textContent = "Pick a run folder or its CSV files first."; return; }',
  '  var runName = folder.length ? (folder[0].webkitRelativePath || "").split("/")[0] : "";',
  '  status.textContent = "Reading " + picked.length + " file(s)...";',
  '  Promise.all(picked.map(function (f) {',
  '    return f.text().then(function (text) { return {name: f.name, text: text}; });',
  '  })).then(function (files) {',
  '    status.textContent = "Importing...";',
  '    google.script.run',
  '      .withSuccessHandler(function (msg) { status.textContent = msg; setTimeout(google.script.host.close, 2500); })',
  '      .withFailureHandler(function (err) { status.textContent = "Failed: " + err.message; })',
  '      .importCsvs(files, runName);',
  '  });',
  '};',
  '</script></body></html>',
].join('\n');
