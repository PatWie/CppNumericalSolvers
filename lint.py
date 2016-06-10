#!/usr/bin/env python
import re
import sys
import getopt
import codecs
import unicodedata


_settings_extensions = set(['cc', 'h', 'cpp', 'cu', 'cuh','hpp', 'cxx'])
_settings_lineWidth  = 150
_settings_extraInfo  = True

_settings_checks = {"studentname":False, 
                    "space indentation":True,
                    "line width":True,
                    "space before {":True,
                    "symbol whitespace":True,
                    "trailing whitespace":True,
                    "function definition whitespace":True,
                    "camelCase":True,
                    "endif comment":True,
                    "global namespaces":True,
                    "multiple empty lines":True
                    }


regExCache = {}
ERROR_COUNTER = 0;

# class bcolors:
#     HEADER = '\033[95m'
#     OKBLUE = '\033[94m'
#     OKGREEN = '\033[92m'
#     WARNING = '\033[93m'
#     FAIL = '\033[91m'
#     ENDC = '\033[0m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'

class RegExMatcher(object):
    def __init__(self, matchstring):
        self.matchstring = matchstring

    def match(self,regExExpr):
      if regExExpr not in regExCache:
        regExCache[regExExpr] = re.compile(regExExpr)
      self.ans = regExCache[regExExpr].match(self.matchstring)
      return bool(self.ans)

    def search(self,regExExpr):
      if regExExpr not in regExCache:
        regExCache[regExExpr] = re.compile(regExExpr)
      self.ans = regExCache[regExExpr].search(self.matchstring)
      return bool(self.ans)

    def findall(self,regExExpr):
      if regExExpr not in regExCache:
        regExCache[regExExpr] = re.compile(regExExpr)
      self.ans = regExCache[regExExpr].finditer(self.matchstring)
      return bool(self.ans)

    def all(self):
      return self.ans
    

    def span(self):
      return self.ans.span()


def lineWidth(line):
  if isinstance(line, unicode):
    width = 0
    for uc in unicodedata.normalize('NFC', line):
      if unicodedata.east_asian_width(uc) in ('W', 'F'):
        width += 2
      elif not unicodedata.combining(uc):
        width += 1
    return width
  else:
    return len(line)

def print_error(filename, linenum, category, message, level=0, line="", char=-1):
  global ERROR_COUNTER
  ERROR_COUNTER = ERROR_COUNTER + 1;
  # level 0: warning
  # level 1: error
  if level == 0:
    sys.stderr.write('(line %s): \033[91m[%s]\033[0m %s\n' % ( linenum, category, message))
  elif level == 1:
    sys.stderr.write('(line %s): \033[91m[%s]\033[0m %s\n' % ( linenum, category, message))
  else:
    sys.stderr.write('(line %s): [%14s] %s\n' % ( linenum, category, message))

  if _settings_extraInfo:
    if char != -1:
      sys.stderr.write('%s %s\n' % (' '*(10+3+3 + len(str(linenum))),line) )
      sys.stderr.write('%s \033[96m%s%s\033[0m\n' % (' '*(10+3+3 + len(str(linenum))),'-'*char,'^') )


def processBraces(filename, lines, error):
  ident = [];
  for line in range(len(lines)):
    for char in range(len(lines[line])):
      if lines[line][char] == '{':
        fullstr = lines[line]
        startwhites = len(fullstr)-len(fullstr.lstrip())
        ident.append( (startwhites, line, char+2) )

      if lines[line][char] == '}':
        fullstr = lines[line]
        startwhites = len(fullstr)-len(fullstr.lstrip())

        if len(ident) == 0:
          error(filename, line, 'ident', 'bracket in line %i at pos %i has no related opening' % (line,char ) ,1, lines[line], char)
          break;
        
        if(ident[len(ident)-1][1] != line):
          if(ident[len(ident)-1][0] != startwhites):
            error(filename, line, 'ident', 'indentation of bracket in line %i does not match indentation of bracket in line %i' % (line,ident[len(ident)-1][1]) )
        del ident[-1]



def processCleanFileLine(filename, file_extension, lines, error, linenum, raw):

  line = lines[linenum]
  rline = raw[linenum]
  # check if line is comment
  m = RegExMatcher(line)
  if m.match(r'\s*//'):
    return
  
  if _settings_checks["space before {"]:
    m = RegExMatcher(line)
    if m.search(r'\S\{'):
      error(filename, linenum, 'braces', 'found no space before {',1 , rline, m.span()[1]-1)
  
  if _settings_checks["symbol whitespace"]:
    m = RegExMatcher(line)
    if m.search(r'[,=]\S'):
      # exclude cases == 
      if not line[m.span()[0]:m.span()[1]] == "==":
        error(filename, linenum, 'whitespace', 'no whitespace after symbol',1 , rline, m.span()[1]-2)
  
  if _settings_checks["function definition whitespace"]:
    m = RegExMatcher(line)
    if m.search(r'[^(for)(while)(if)(else)\n&\*/\+\-\|=\,]+\s+\('):
      # exclude special case with only whitespaces
      mm = RegExMatcher(line[0:m.span()[1]])
      if not mm.search(r'^\s+\('):
        if not mm.search(r'return\s\('):
          error(filename, linenum, 'whitespace', 'found whitespace before (',1 , rline, m.span()[1]-1)
  
  if _settings_checks["camelCase"]:
    m = RegExMatcher(line)
    if m.search(r'\w+\s*\s[A-Z]+\w*\(.*\{'):
      error(filename, linenum, 'capitalization', 'found function without camelCase',1 , rline, m.span()[1]-1)

  if _settings_checks["global namespaces"]:
    m = RegExMatcher(line)
    if m.search(r'using\s+namespace'):
      error(filename, linenum, 'style', 'do not use global namespaces',1 , rline, m.span()[1]-1)

def processFileLine(filename, file_extension, lines, error, linenum):

  line = lines[linenum]
  # check if line is comment
  m = RegExMatcher(line)
  if m.match(r'\s*//'):
    return
  if _settings_checks["space indentation"]:
    m = RegExMatcher(line)
    if m.search('\t'):
      error(filename, linenum, 'tab', 'use spaces instead of tabs',1 , line, m.span()[1]-1)
  
  if _settings_checks["line width"]:
    if lineWidth(line) > _settings_lineWidth:
      error(filename, linenum, 'line_length', 'lines should be <= %i characters long' % _settings_lineWidth)
  
  if _settings_checks["trailing whitespace"]:
    m = RegExMatcher(line)
    if m.search(r'\s+$'):
      error(filename, linenum, 'whitespace', 'found trailing whitespace',1 , line, m.span()[1]-1)
  
  if _settings_checks["endif comment"]:
    m = RegExMatcher(line)
    if m.search(r'#endif\s*$'):
      error(filename, linenum, 'comment', 'found endif without comment',1 , line, m.span()[1]-1)


def processCleanFileContent(filename, file_extension, lines, ignore_lines, error, raw):
  lines = (['// marker so line numbers and indices both start at 1'] + lines +
           ['// marker so line numbers end in a known way'])
  raw = (['// marker so line numbers and indices both start at 1'] + raw +
           ['// marker so line numbers end in a known way'])
  for line in range(len(lines)):
    if line not in ignore_lines:
      processCleanFileLine(filename, file_extension, lines, error, line, raw)

def processFileContent(filename, file_extension, lines, ignore_lines, error):
  # test if file is ending with empty lines
  empty_lines = 0;
  for i in range(len(lines)):
    if lines[len(lines)-i-1] == "":
      empty_lines = empty_lines + 1;
    else:
      break

  if  empty_lines == 0:
    error(filename, len(lines)-empty_lines, 'empty', 'file ends without empty lines')


  lines = (['// marker so line numbers and indices both start at 1'] + lines +
           ['// marker so line numbers end in a known way'])
  for line in range(len(lines)):
    if line not in ignore_lines:
      processFileLine(filename, file_extension, lines, error, line)

  processBraces(filename, lines, error)

# http://stackoverflow.com/questions/241327/python-snippet-to-remove-c-and-c-comments
def comment_remover(text):
    
    def blotOutNonNewlines( strIn ) :  # Return a string containing only the newline chars contained in strIn
        return "" + ("\n" * strIn.count('\n'))

    def replacer( match ) :
        s = match.group(0)
        if s.startswith('/'):  # Matched string is //...EOL or /*...*/  ==> Blot out all non-newline chars
            return "@#@"+blotOutNonNewlines(s)
        else:                  # Matched string is '...' or "..."  ==> Keep unchanged
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )

    raw = re.sub(pattern, replacer, text)
    raw = re.sub(r'[ ]*@#@','',raw)
    return raw;



def processFile(filename):
  error_collection = [];

  def error_handler(filename, linenum, category, message, level=0, line="", char=-1):
    error_collection.append( (filename, linenum, category, message, level, line, char ) )

  file_extension = filename[filename.rfind('.') + 1:]

  if not file_extension in _settings_extensions:
    return;

  raw_source = codecs.open(filename, 'r', 'utf8', 'replace').read()
  lines = raw_source.split('\n')
  lf_lines = []
  crlf_lines = []
  for linenum in range(len(lines) - 1):
      if lines[linenum].endswith('\r'):
        lines[linenum] = lines[linenum].rstrip('\r')
        crlf_lines.append(linenum + 1)
      else:
        lf_lines.append(linenum + 1)

  # check each line
  if lf_lines and crlf_lines :
      for linenum in crlf_lines:
        error_handler(filename, linenum, 'newline', 'use \\n here')

  # check authorname
  if _settings_checks["studentname"]:
    m = RegExMatcher(lines[0])
    if not m.match(r'// @student:\s\w+[\s*\w*]*'):
      error_handler(filename, 1, 'content', 'failed finding name of student in first line. Correct: "// @student: foo"',1)

  # check double newline
  if _settings_checks["multiple empty lines"]:
    m = RegExMatcher(raw_source)
    if m.findall(r'(\n[ ]*){3,}'):
      for mm in m.all():
        error_handler(filename, len(raw_source[:mm.span()[1]].split('\n'))-1, 'empty', 'too many empty new lines (count %i)' % (-2+len(raw_source[mm.span()[0]:mm.span()[1]].split('\n'))),1)
      

  # get lines to ignore
  ignore_lines = [];
  pattern = re.compile(r'nolintnextline')
  m.findall(pattern)
  for mm in m.all():
    ignore_lines.append(len(raw_source[:mm.span()[1]].split('\n')) + 1)
  pattern = re.compile(r'nolint')
  m.findall(pattern)
  for mm in m.all():
    ignore_lines.append(len(raw_source[:mm.span()[1]].split('\n')))


  processFileContent(filename, file_extension, lines, ignore_lines, error_handler)
  # clean lines (remove multi line comment)
  lines  = raw_source.split('\n')
  clines = comment_remover(raw_source).split('\n')
  processCleanFileContent(filename, file_extension, clines, ignore_lines, error_handler, lines)

  if len(error_collection):
    sys.stderr.write('\033[1m\033[93m%10s\033[0m:\n' % ( filename))
    for e in error_collection:
      print_error(*e)
    



def main():
  filenames = sys.argv[1:];
  for filename in filenames:
    processFile(filename)

  sys.stderr.write("\n\n");
  if ERROR_COUNTER == 0:
    #sys.stderr.write("\033[92m NO LINT ERRORS! \033[0m\n");
    sys.exit(ERROR_COUNTER)
  else:
    # if ERROR_COUNTER == 1:
    #   sys.stderr.write("\033[91m FOUND %i LINT ERROR! \033[0m\n" % ERROR_COUNTER);
    # else:
    #   sys.stderr.write("\033[91m FOUND %i LINT ERRORS! \033[0m\n" % ERROR_COUNTER);
    sys.exit(ERROR_COUNTER)


if __name__ == '__main__':
  main()
