/**
 * This file is a Google Apps Script that imports run CSV files into the open
 * Google Sheet: one tab per run, named after its map and video
 * ("distractor_floorplan_E - S-19_Elevator"). The run's summary.csv,
 * room_order.csv and corner_motion.csv sit side by side on that tab, each
 * converted to its own Google Sheets table (and any other CSV in the run
 * folder becomes one more table after them).
 *
 * Setup, once per spreadsheet:
 *   1. In the Google Sheet: Extensions > Apps Script.
 *   2. Replace the editor's contents with this whole file and save.
 *   3. In the editor's left bar: Services (+) > Google Sheets API > Add.
 *      This is what turns the CSVs into real tables; without it they're
 *      imported as plain striped ranges instead.
 *   4. Reload the sheet. A "Tactile Maps" menu appears.
 *
 * Use: Tactile Maps > Import run CSVs..., then pick the whole data/ folder
 * (every run in it), one run folder (data/<date_time>__<map>__<video>/), or
 * just some CSV files. Importing a run again replaces its tab; other tabs
 * are left alone. If the folder holds several runs of the same map and
 * video, the newest one is the one kept. The first time, Google asks you
 * to authorize the script for this spreadsheet.
 *
 * The CSVs come from data_export.py: plain tables, header row first, times
 * in seconds. Numbers are written as numbers; everything else as plain text,
 * so a cell like "(8.5, 11)" isn't reinterpreted by Sheets. Runs from before
 * the CSV split (one data.csv, no summary.csv) are skipped and listed.
 */

// CSV file name -> table name, in left-to-right order on the tab. Other
// CSVs follow, named from their file name ("finger_speed.csv" -> "Finger Speed").
var TABLES = [
  ['summary.csv', 'Summary'],
  ['room_order.csv', 'Room Order'],
  ['corner_motion.csv', 'Corner Motion'],
];

var RUN_NAME = /^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}__(.+)__(.+)$/;   // <date_time>__<map>__<video>
var NUMBER = /^-?\d+(\.\d+)?$/;
var GAP_COLUMNS = 1;   // empty columns between two tables

function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('Tactile Maps')
    .addItem('Import run CSVs...', 'showImportDialog')
    .addItem('Delete all tabs...', 'deleteAllTabs')
    .addToUi();
}

/** Deletes every tab after asking. A spreadsheet must keep one tab, so it's
 * left with a single empty "Sheet1" (which the next import drops). */
function deleteAllTabs() {
  var ui = SpreadsheetApp.getUi();
  var answer = ui.alert('Delete all tabs',
    'Delete every tab in this spreadsheet? This cannot be undone.', ui.ButtonSet.YES_NO);
  if (answer !== ui.Button.YES) return;
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var blank = ss.insertSheet('Empty ' + Date.now());
  ss.getSheets().forEach(function (s) {
    if (s.getSheetId() !== blank.getSheetId()) ss.deleteSheet(s);
  });
  blank.setName('Sheet1');
}

function showImportDialog() {
  var html = HtmlService.createHtmlOutput(DIALOG_HTML).setWidth(460).setHeight(300);
  SpreadsheetApp.getUi().showModalDialog(html, 'Import run CSVs');
}

/**
 * Called from the dialog once per run. run: the run folder's name ("" for
 * loose CSV files); files: [{name: "summary.csv", text: "..."}].
 * Returns {tab, tables: true if real tables were made, why: the reason if not}.
 */
function importRun(run, files) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var label = tabLabel(run);
  var csvs = files.filter(function (f) { return /\.csv$/i.test(f.name); })
    .sort(function (a, b) { return tableOrder(a.name) - tableOrder(b.name); });

  // A fresh tab, in the old one's place (or in name order), so a re-import
  // never leaves stale cells or tables behind.
  var old = ss.getSheetByName(label);
  var position = old ? old.getIndex() - 1 : ss.getSheets().filter(function (s) {
    return s.getName() < label;
  }).length;
  var sheet = ss.insertSheet('Importing ' + Date.now(), position);
  if (old) ss.deleteSheet(old);
  sheet.setName(label);

  // Write each CSV, left to right.
  var placed = [], col = 1;
  csvs.forEach(function (f) {
    var values = toValues(f.text);
    if (!values) return;
    var height = values.length, width = values[0].length;
    if (sheet.getMaxRows() < height) sheet.insertRowsAfter(sheet.getMaxRows(), height - sheet.getMaxRows());
    if (sheet.getMaxColumns() < col + width - 1) {
      sheet.insertColumnsAfter(sheet.getMaxColumns(), col + width - 1 - sheet.getMaxColumns());
    }
    // Text columns (any non-number cell below the header) are set to plain
    // text first, so Sheets keeps "(8.5, 11)" or a room name exactly as is.
    for (var c = 0; c < width; c++) {
      var isText = false;
      for (var r = 1; r < height && !isText; r++) isText = typeof values[r][c] === 'string' && values[r][c] !== '';
      if (isText) sheet.getRange(1, col + c, height, 1).setNumberFormat('@');
    }
    var range = sheet.getRange(1, col, height, width);
    range.setValues(values);
    sheet.getRange(1, col, 1, width).setFontWeight('bold');
    placed.push({name: tableName(f.name), range: range});
    col += width + GAP_COLUMNS;
  });
  if (!placed.length) throw new Error('No data in the CSV files for ' + label + '.');

  var why = makeTables(ss, sheet, label, placed);
  sheet.setFrozenRows(1);
  sheet.autoResizeColumns(1, col - 1 - GAP_COLUMNS);
  return {tab: label, tables: !why, why: why};
}

/** Called from the dialog after the last run: drops the empty "Sheet1" a
 * new spreadsheet starts with and shows the first tab. */
function finishImport() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  ss.getSheets().forEach(function (s) {
    if (ss.getSheets().length > 1 && s.getLastRow() === 0 && s.getLastColumn() === 0) ss.deleteSheet(s);
  });
  ss.setActiveSheet(ss.getSheets()[0]);
}

/**
 * Turns each placed range into a Google Sheets table through the Sheets
 * API (the "Google Sheets API" advanced service). Returns "" on success;
 * otherwise stripes the ranges instead and returns why tables weren't made.
 */
function makeTables(ss, sheet, label, placed) {
  var why = 'the Google Sheets API service is not added (Apps Script editor > Services > + > Google Sheets API > Add)';
  if (typeof Sheets !== 'undefined') {
    try {
      SpreadsheetApp.flush();
      var sheetId = sheet.getSheetId();
      var requests = placed.map(function (p) {
        var r = p.range;
        return {addTable: {table: {
          name: apiName(p.name + ' ' + label),
          range: {sheetId: sheetId,
                  startRowIndex: r.getRow() - 1, endRowIndex: r.getLastRow(),
                  startColumnIndex: r.getColumn() - 1, endColumnIndex: r.getLastColumn()},
        }}};
      });
      Sheets.Spreadsheets.batchUpdate({requests: requests}, ss.getId());
      return '';
    } catch (e) {
      why = 'the Sheets API refused: ' + (e && e.message || e);
    }
  }
  console.warn('Tables not created for ' + label + ': ' + why);
  placed.forEach(function (p) { p.range.applyRowBanding(SpreadsheetApp.BandingTheme.LIGHT_GREY, true, false); });
  return why;
}

/** CSV text -> rows of cell values, all the same width; null if empty.
 * A header-only CSV gets one blank row, since a table needs a body. */
function toValues(text) {
  var rows = Utilities.parseCsv(text).filter(function (r) { return r.join('') !== ''; });
  if (!rows.length) return null;
  var width = Math.max.apply(null, rows.map(function (r) { return r.length; }));
  var values = rows.map(function (r, i) {
    var row = i === 0 ? r.map(function (h) { return String(h).trim(); }) : r.map(cellValue);
    while (row.length < width) row.push('');
    return row;
  });
  if (values.length === 1) values.push(values[0].map(function () { return ''; }));
  return values;
}

function cellValue(cell) {
  var s = String(cell).trim();
  return NUMBER.test(s) ? Number(s) : s;
}

/** Run folder name -> tab name: "<map> - <video>". */
function tabLabel(run) {
  var m = RUN_NAME.exec(run || '');
  var label = m ? m[1] + ' - ' + m[2] : (run || 'Imported CSVs');
  return label.replace(/[\[\]*?\/\\:]/g, '_').replace(/^'+|'+$/g, '').slice(0, 100);
}

function tableName(fileName) {
  var base = fileName.split('/').pop();
  for (var i = 0; i < TABLES.length; i++) {
    if (TABLES[i][0] === base.toLowerCase()) return TABLES[i][1];
  }
  return base.replace(/\.csv$/i, '').split(/[_\s-]+/).map(function (w) {
    return w.charAt(0).toUpperCase() + w.slice(1);
  }).join(' ');
}

function tableOrder(fileName) {
  var base = fileName.split('/').pop().toLowerCase();
  for (var i = 0; i < TABLES.length; i++) {
    if (TABLES[i][0] === base) return i;
  }
  return TABLES.length;
}

/** A table name the Sheets API accepts and that's unique per tab:
 * letters, digits and underscores ("Summary_distractor_floorplan_E_S_19_Elevator"). */
function apiName(s) {
  var name = s.replace(/[^A-Za-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
  return (/^[A-Za-z]/.test(name) ? name : 'T_' + name).slice(0, 100);
}

// Note: the dialog's script runs inside a function, and its status line is
// "statusEl" -- a global named "status" would be window.status, which only
// holds strings, so the dialog would silently never show anything.
var DIALOG_HTML = [
  '<!doctype html><html><head><base target="_top"><style>',
  'body{font:14px Arial,sans-serif;margin:0 4px}',
  'label{display:block;margin:10px 0 4px;font-weight:bold}',
  '.hint{color:#666;font-size:12px}',
  '#status{margin-top:14px;min-height:4em}',
  'button{margin-top:12px;padding:6px 16px}',
  '</style></head><body>',
  '<label for="folder">Folder</label>',
  '<input type="file" id="folder" webkitdirectory>',
  '<div class="hint">The whole data folder (every run), or one run folder.</div>',
  '<label for="files">or CSV files from one run</label>',
  '<input type="file" id="files" multiple accept=".csv,text/csv">',
  '<br><button id="go">Import</button>',
  '<div id="status"></div>',
  '<script>',
  '(function () {',
  '  var KNOWN = ["summary.csv", "room_order.csv", "corner_motion.csv"];',
  '  var statusEl = document.getElementById("status");',
  '  var go = document.getElementById("go");',
  '  function say(msg) { statusEl.textContent = msg; }',
  '  function call(fn, args) {',
  '    return new Promise(function (ok, fail) {',
  '      var runner = google.script.run.withSuccessHandler(ok).withFailureHandler(fail);',
  '      runner[fn].apply(runner, args);',
  '    });',
  '  }',
  '  go.onclick = function () {',
  '    var folder = document.getElementById("folder").files;',
  '    var picked = Array.prototype.slice.call(folder.length ? folder : document.getElementById("files").files)',
  '      .filter(function (f) { return /\\.csv$/i.test(f.name); });',
  '    if (!picked.length) { say("Pick a folder or some CSV files first (no .csv files found)."); return; }',
  // Group by the folder each CSV is in: one group = one run.
  '    var runs = {};',
  '    picked.forEach(function (f) {',
  '      var parts = (f.webkitRelativePath || "").split("/");',
  '      var run = parts.length >= 2 ? parts[parts.length - 2] : "";',
  '      (runs[run] = runs[run] || []).push(f);',
  '    });',
  '    var names = Object.keys(runs).sort(), todo = [], old = [];',
  '    names.forEach(function (run) {',
  '      var known = runs[run].some(function (f) { return KNOWN.indexOf(f.name.toLowerCase()) >= 0; });',
  '      if (known) todo.push(run); else old.push(run || "(loose files)");',
  '    });',
  '    if (!todo.length) { say("No run CSVs found (summary.csv, room_order.csv, corner_motion.csv)."); return; }',
  '    go.disabled = true;',
  '    var done = 0, tabs = {}, plain = 0, why = "";',
  // One server call per run, oldest first, so the newest run of a map + video wins its tab.
  '    todo.reduce(function (chain, run) {',
  '      return chain.then(function () {',
  '        say("Importing run " + (done + 1) + " of " + todo.length + "...");',
  '        return Promise.all(runs[run].map(function (f) {',
  '          return f.text().then(function (text) { return {name: f.name, text: text}; });',
  '        })).then(function (files) { return call("importRun", [run, files]); })',
  '          .then(function (res) { done++; tabs[res.tab] = true; if (!res.tables) { plain++; why = res.why; } });',
  '      });',
  '    }, Promise.resolve()).then(function () {',
  '      return call("finishImport", []);',
  '    }).then(function () {',
  '      var n = Object.keys(tabs).length;',
  '      var msg = "Imported " + done + " run(s) into " + n + " tab(s).";',
  '      if (done > n) msg += " " + (done - n) + " older run(s) of the same map and video were replaced by newer ones.";',
  '      if (old.length) msg += " Skipped " + old.length + " old-format run(s) with only data.csv: " + old.join(", ") + ".";',
  '      if (plain) msg += " Tables were not created on " + plain + " tab(s) because " + why + ".";',
  '      say(msg);',
  '    }).catch(function (err) {',
  '      say("Failed after " + done + " of " + todo.length + " run(s): " + (err && err.message || err));',
  '    }).then(function () { go.disabled = false; });',
  '  };',
  '})();',
  '</script></body></html>',
].join('\n');
