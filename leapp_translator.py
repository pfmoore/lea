'''
--------------------------------------------------------------------------------

    leapp_translator.py

--------------------------------------------------------------------------------
Copyright 2013-2016 Pierre Denis

This file is part of Lea.

Lea is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Lea is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.Leapp

You should have received a copy of the GNU Lesser General Public License
along with Lea.  If not, see <http://www.gnu.org/licenses/>.
--------------------------------------------------------------------------------
'''

import string
from .prob_fraction import ProbFraction
from .toolbox import zip

# leapp internal configuration parameters, language-dependant
IDENTIFIER_CHARACTERS = string.ascii_letters + string.digits + '_'
LITERAL_CHARACTERS = IDENTIFIER_CHARACTERS + '#'
STRING_SEPARATORS = ('"',"'")
PRINT_SUFFIXES = '%/.-' + string.whitespace

CONTINUATION_LINE_CHARS2 = ','

class LeappTranslator(object):

    ''' 
    LeappTranslator provides static methods to translate a multiline 
    leapp statement into Python statement.
    The entry-point method is getTarget00.
    
    IMPORTANT NOTE:
    The present implementation of LeappTranslator does NOT use usual
    recommanded parsing techniques nor specialized modules.
    For this reason, it is admittedly VERY convoluted!
    One of the rationale is that Leapp does NOT intend to be a true progamming
    language; it merely adds some syntactic sugar on top of Python, to ease
    the usage of Lea. The Python fragments interleaved in Leapp syntax are
    not "understood" by Leapp translator, which simply copy them on output.
    This brutal approach is inadequate for the maintenance and is unsatisfying
    for the Leapp user who receives poor syntax error diagnosis.
    
    For these reasons, the present implementation is sentenced to a deep refactoring!

    '''
    
    class Error(Exception):
        pass

    @staticmethod
    def unindent(line):
        unindentedLine = line.strip()
        if len(unindentedLine) == 0:
            indentStr = ''
        else:   
            indentStr = line[:line.index(unindentedLine)]
        return (indentStr,unindentedLine)

    @staticmethod
    def getTarget00(rMultilineStatement):
        ''' 
        '''
        leaStatementLines = rMultilineStatement.split('\n')
        # merge lines that require it
        leaStatementLines1 = []
        leaStatementLineParts = []
        deindent = False
        for leaStatementLine in leaStatementLines:
            leaStatementLine = leaStatementLine.rstrip()
            if len(leaStatementLine) == 0:
                if len(leaStatementLineParts) > 0:
                    if len(leaStatementLineParts) > 0:
                        leaStatementLines1.append(''.join(leaStatementLineParts))
                        leaStatementLineParts = []
                    leaStatementLines1.append('')
            else:
                if deindent:
                    leaStatementLine = leaStatementLine.lstrip()           
                lastChar = leaStatementLine[-1]
                if lastChar == '\\':
                    leaStatementLine = leaStatementLine[:-1]
                leaStatementLineParts.append(leaStatementLine)
                if len(leaStatementLine) <= 1 or lastChar not in CONTINUATION_LINE_CHARS2:
                    leaStatementLines1.append(''.join(leaStatementLineParts))
                    leaStatementLineParts = []
                    deindent = False
                else:
                    deindent = True
        if len(leaStatementLineParts) > 0:
            leaStatementLines1.append(''.join(leaStatementLineParts))
        pStatementLines = []
        for leaStatementLine in leaStatementLines1:
            (identStr,unindentedLeaStatementLine) = LeappTranslator.unindent(leaStatementLine)
            pStatementLines.append(identStr+LeappTranslator.getTarget0(unindentedLeaStatementLine))
        return '\n'.join(pStatementLines)

    @staticmethod
    def getTarget0(sourceFragment):
        # assume sourceFragment is stripped both sides
        lenSourceFragment = len(sourceFragment)
        headTarget = ''
        tailTarget = ''
        if lenSourceFragment >= 1 and sourceFragment[0] == ':':
            # print statement
            if len(sourceFragment) <= 1:
                # empty
                return "print('')"
            prefixLength = 1
            printTypeCode = ' '
            secondChar = sourceFragment[1]
            if secondChar in PRINT_SUFFIXES:
                printTypeCode = secondChar
                prefixLength += 1
            if len(sourceFragment) >= 3:
                thirdChar = sourceFragment[2]
                if thirdChar == '-':
                    printTypeCode += '-'
                    prefixLength += 1
            sourceFragment = sourceFragment[prefixLength:]
            headTarget = 'print(('
            tailTarget = ''
            if printTypeCode == '.':
                tailTarget += ').asFloat())'
            elif printTypeCode == '%':
                tailTarget += ').asPct())'
            elif '-' in printTypeCode:
                tailTarget = '%s).asString("%s"))' % (tailTarget,printTypeCode)
            else:
                tailTarget += '))'
        return headTarget + LeappTranslator.getTarget0b(sourceFragment) + tailTarget  
        
    @staticmethod
    def getTarget0b(sourceFragment):
        sourceFragment = sourceFragment.strip()
        isBool = len(sourceFragment) >= 1 and (sourceFragment[0] == '@')
        if isBool:
            sourceFragment = sourceFragment[1:]
        target = LeappTranslator.getTarget1(sourceFragment)
        if isBool:
            target = '(' + target + ').p(True)'
        return target
        
    @staticmethod
    def getTarget1(sourceFragment):
        sourceFragments = []
        currentStringDelim = None
        chars = []
        for c in sourceFragment:
            if currentStringDelim is None:
                if c == '#':
                    break
                if c in STRING_SEPARATORS:
                    currentStringDelim = c
                    if len(chars) > 0:
                        sourceFragments.append(''.join(chars))
                    chars = []
                chars.append(c)
            elif c == currentStringDelim:
                currentStringDelim = None
                chars.append(c)
                sourceFragments.append(''.join(chars))
                chars = []
            else:
                chars.append(c)
        if len(chars) > 0:
            sourceFragments.append(''.join(chars))
        extractedStrings = ['']
        unstringedFragmentParts = []
        for sourceFragment in sourceFragments:
            if sourceFragment[0] in STRING_SEPARATORS:
                extractedStrings.append(sourceFragment)
                unstringedFragmentParts.append('#')
            else:
                unstringedFragmentParts.append(sourceFragment)
        unstringedFragment = ''.join(unstringedFragmentParts)
        unstringedTargetFragment = LeappTranslator.getTarget(unstringedFragment)
        fragmentParts = []
        targetFragment = ''.join(s1+s2 for (s1,s2) in zip(extractedStrings,unstringedTargetFragment.split('#')))
        return targetFragment
    
    @staticmethod
    def convertFraction(fractionExpression):
        fractionExpression = fractionExpression.strip()
        if '/' in fractionExpression or fractionExpression.endswith('%') \
         or fractionExpression.startswith('0') or fractionExpression.startswith('.'):
            probFraction = ProbFraction(fractionExpression)
            return "%s,%s" % (probFraction.numerator,probFraction.denominator)
        return fractionExpression

    @staticmethod
    def smartSplit(fragment,sep):
        levelParentheses = 0
        levelBraces = 0
        levelBrackets = 0
        parts = []
        chars = []
        chars2 = []
        lenSep = len(sep)
        nbMatchChars = 0
        sepC = sep[0]
        for c in fragment:
            if c == '(':
                levelParentheses += 1
            elif c == ')':
                levelParentheses -= 1
            elif c == '{':
                levelBraces += 1
            elif c == '}':
                levelBraces -= 1
            elif c == '[':
                levelBrackets += 1
            elif c == ']':
                levelBrackets -= 1
            if c == sepC and levelParentheses+levelBraces+levelBrackets == 0:
                chars2.append(c)
                nbMatchChars += 1
                if nbMatchChars == lenSep:
                    parts.append(''.join(chars))
                    del chars[:]
                    del chars2[:]
                    nbMatchChars = 0
                sepC = sep[nbMatchChars]
            else:
                if nbMatchChars > 0:
                    nbMatchChars = 0
                    chars += chars2
                    del chars2[:] 
                    sepC = sep[0]    
                chars.append(c)
            if levelParentheses<0 or levelBraces<0 or levelBrackets<0:
                raise LeappTranslator.Error('missing opening delimiter')
        if levelParentheses>0 or levelBraces>0 or levelBrackets>0:
            raise LeappTranslator.Error('missing closing pairing')
        if len(chars) > 0:
            parts.append(''.join(chars))
        return parts        

    @staticmethod
    def treatProbWeightExpression(dictExpression):        
        if ':' in dictExpression:
            # dictionary literal: enclose it in brackets
            distributionItems = tuple(LeappTranslator.smartSplit(item,':') for item in LeappTranslator.smartSplit(dictExpression,','))
            probFractions = tuple(ProbFraction(probWeightString) for (valStr,probWeightString) in distributionItems)
            probWeights = ProbFraction.getProbWeights(probFractions)
            newDictExpression = ','.join('(%s,%d)'%(LeappTranslator.getTarget(valStr),probWeight) \
                                 for ((valStr,probWeightString),probWeight) in zip(distributionItems,probWeights))
            return 'Lea.fromValFreqs(%s)' % newDictExpression
        # dictionary variable: expand items
        return 'Alea.fromValFreqsDictGen(%s)' % dictExpression
        
    @staticmethod
    def treatCPTExpression(cptExpression):
        if '->' in cptExpression:
            # CPT literal
            def f(condExpr):
                if condExpr.strip() == '_':
                    return None
                return condExpr    
            cptItems = tuple(LeappTranslator.smartSplit(item,'->') for item in LeappTranslator.smartSplit(cptExpression,','))
            newCPTExpression = ','.join('(%s,%s)'%(f(condExpr),distribExpr) for (condExpr,distribExpr) in cptItems)
            return newCPTExpression
        return cptExpression
        
    @staticmethod
    def getTarget(sourceFragment):
        targetFragment = sourceFragment
        targetFragment = LeappTranslator.parse(targetFragment,'?*','(',')','Lea.cprod(%s)')
        targetFragment = LeappTranslator.parse(targetFragment,'?!' ,'(',')','Lea.buildCPT(*(%s,))',LeappTranslator.treatCPTExpression)
        targetFragment = LeappTranslator.parse(targetFragment,'!!' ,'(',')','.revisedWithCPT(*(%s,))',LeappTranslator.treatCPTExpression)
        targetFragment = LeappTranslator.parse(targetFragment,'?' ,'(',')','Lea.fromVals(*(%s))')
        targetFragment = LeappTranslator.parse(targetFragment,'?:','(',')','Lea.boolProb(%s)',LeappTranslator.convertFraction)
        targetFragment = LeappTranslator.parse(targetFragment,'?' ,'{','}','%s',LeappTranslator.treatProbWeightExpression)
        targetFragment = LeappTranslator.parse(targetFragment,'!' ,'','','.given(%s)')
        targetFragment = LeappTranslator.parse(targetFragment,'$_' ,'(',')','.randomDraw(%s)')
        targetFragment = LeappTranslator.parse(targetFragment,'$' ,'(',')','.random(%s)')
        targetFragment = LeappTranslator.parseSingle(targetFragment,'$','.random()')
        targetFragment = LeappTranslator.parseSingle(targetFragment,'.@','.new()')
        targetFragment = LeappTranslator.parse(targetFragment,'@' ,'(',')','.p(%s)')
        targetFragment = LeappTranslator.parseIsolatedAt(targetFragment)
        targetFragment = LeappTranslator.parseSingle(targetFragment,'@','.p(True)')
        (head,timesArgs,identifier,tail) = LeappTranslator.parseIdentifier(targetFragment,'?')
        if head is not None:
            if identifier is None:
                raise LeappTranslator.Error('missing identifier')
            timesExpr = ''
            if timesArgs is not None:
                timesExpr = '.times(%s)'% timesArgs
            isFunc = tail.lstrip().startswith('(')
            if isFunc:
                targetFragment = head + LeappTranslator.parse(tail,'','(',')',('Flea.build(%s,(%%s,))'%identifier)+timesExpr)
            else:
                targetFragment = head + identifier
                if timesExpr:
                    targetFragment += timesExpr
                else:
                    targetFragment += '.clone()'
                targetFragment += LeappTranslator.getTarget(tail)
        return targetFragment

    @staticmethod
    def parseIsolatedAt(sourceFragment):
        splitResult = sourceFragment.split('@',1)
        if len(splitResult) == 1:
            return sourceFragment
        (head,tail) = splitResult
        if len(tail) == 0 or tail[0] not in LITERAL_CHARACTERS:
            return sourceFragment
        splitResult2 = tail.split(None,1)
        if len(splitResult2) == 2:
            (body,tail2) = splitResult2
        else:
            body = tail
            tail2 = ''
        return '%s.p(%s) %s' % (head,LeappTranslator.getTarget(body),LeappTranslator.getTarget(tail2))

    @staticmethod
    def parseSingle(sourceFragment,prefixToken,targetBody):
        splitResult = sourceFragment.split(prefixToken,1)
        if len(splitResult) == 1:
            return sourceFragment
        (head,tail) = splitResult
        return head + targetBody + LeappTranslator.getTarget(tail)

    @staticmethod
    def parse(sourceFragment,prefixToken,openToken,closeToken,targetBodyTemplate,treatFunc=None):
        special = openToken == ''
        if prefixToken == '':
            (head1,tail1) = ('',sourceFragment)
        else:
            splitResult1 = sourceFragment.split(prefixToken,1)
            if len(splitResult1) == 1:
                return sourceFragment
            (head1,tail1) = splitResult1
            #head1 = head1.rstrip()
        if special:
            if len(tail1) > 0 and tail1[0] == '=':
                # ignore Python inequality operator '!='
                return head1 + '!' + LeappTranslator.parse(tail1,prefixToken,openToken,closeToken,targetBodyTemplate)
            (head2,tail2) = ('',tail1)
            (openToken,closeToken) = ('(',')')
        else:
            splitResult2 = tail1.split(openToken,1)
            if len(splitResult2) == 1:
                return sourceFragment
            (head2,tail2) = splitResult2
        if len(head2.lstrip()) > 0:
            return sourceFragment
        bodyChars = []
        tailIter = iter(tail2)
        level = 1
        for c in tailIter:
            if c == openToken:
                level += 1
            elif c == closeToken:
                level -= 1
                if level == 0:
                    if special:
                        bodyChars.append(closeToken)
                    break
            bodyChars.append(c)
        if level > 0 and not special:
            raise LeappTranslator.Error("opening '%s' but missing '%s'"%(prefixToken+openToken,closeToken))
        body = ''.join(bodyChars)
        tail = ''.join(tailIter)
        if treatFunc is not None:
            body = treatFunc(body)
        targetBody = LeappTranslator.getTarget(body)
        return head1 + targetBodyTemplate % targetBody + LeappTranslator.getTarget(tail)
        
    @staticmethod
    def getIdentifier(sourceFragment):
        identifierChars = []
        tailIter = iter(sourceFragment)
        tail = ''
        for c in tailIter:
            if c not in IDENTIFIER_CHARACTERS:
                tail = c
                break
            identifierChars.append(c)
        identifier = ''.join(identifierChars)
        tail += ''.join(tailIter)
        return (identifier,tail)

    @staticmethod
    def parseIdentifier(sourceFragment,prefixToken):
        (head,timesArgs,identifier,tail) = (None,None,None,None)
        splitResult = sourceFragment.split(prefixToken,1)
        if len(splitResult) == 2:
            (head,tail1) = splitResult
            tail1 = tail1.lstrip()
            if len(tail1) > 0:
                firstChar = tail1[0]
                if firstChar == '[':
                    splitResult = tail1[1:].split(']',1)
                    if len(splitResult) == 2:
                        (timesArgs,tail1) = splitResult
                        tail1 = tail1.lstrip()
            if len(tail1) > 0:
                firstChar = tail1[0]
                if firstChar in string.ascii_letters:
                    (identifier,tail) = LeappTranslator.getIdentifier(tail1)
        return (head,timesArgs,identifier,tail)

