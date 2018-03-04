'''
--------------------------------------------------------------------------------

    leapp_translator.py

--------------------------------------------------------------------------------
Copyright 2013-2018 Pierre Denis

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
    The entry-point method is get_target00.
    
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
        unindented_line = line.strip()
        if len(unindented_line) == 0:
            indent_str = ''
        else:   
            indent_str = line[:line.index(unindented_line)]
        return (indent_str,unindented_line)

    @staticmethod
    def get_target00(r_multiline_statement):
        ''' 
        '''
        lea_statement_lines = r_multiline_statement.split('\n')
        # merge lines that require it
        lea_statement_lines1 = []
        lea_statement_line_parts = []
        deindent = False
        for lea_statement_line in lea_statement_lines:
            lea_statement_line = lea_statement_line.rstrip()
            if len(lea_statement_line) == 0:
                if len(lea_statement_line_parts) > 0:
                    if len(lea_statement_line_parts) > 0:
                        lea_statement_lines1.append(''.join(lea_statement_line_parts))
                        lea_statement_line_parts = []
                    lea_statement_lines1.append('')
            else:
                if deindent:
                    lea_statement_line = lea_statement_line.lstrip()           
                last_char = lea_statement_line[-1]
                if last_char == '\\':
                    lea_statement_line = lea_statement_line[:-1]
                lea_statement_line_parts.append(lea_statement_line)
                if len(lea_statement_line) <= 1 or last_char not in CONTINUATION_LINE_CHARS2:
                    lea_statement_lines1.append(''.join(lea_statement_line_parts))
                    lea_statement_line_parts = []
                    deindent = False
                else:
                    deindent = True
        if len(lea_statement_line_parts) > 0:
            lea_statement_lines1.append(''.join(lea_statement_line_parts))
        p_statement_lines = []
        for lea_statement_line in lea_statement_lines1:
            (ident_str,unindented_lea_statement_line) = LeappTranslator.unindent(lea_statement_line)
            p_statement_lines.append(ident_str+LeappTranslator.get_target0(unindented_lea_statement_line))
        return '\n'.join(p_statement_lines)

    @staticmethod
    def get_target0(source_fragment):
        # assume source_fragment is stripped both sides
        len_source_fragment = len(source_fragment)
        head_target = ''
        tail_target = ''
        if len_source_fragment >= 1 and source_fragment[0] == ':':
            # print statement
            if len(source_fragment) <= 1:
                # empty
                return "print('')"
            prefix_length = 1
            print_type_code = ' '
            second_char = source_fragment[1]
            if second_char in PRINT_SUFFIXES:
                print_type_code = second_char
                prefix_length += 1
            if len(source_fragment) >= 3:
                third_char = source_fragment[2]
                if third_char == '-':
                    print_type_code += '-'
                    prefix_length += 1
            source_fragment = source_fragment[prefix_length:]
            head_target = 'print(('
            tail_target = ''
            if print_type_code == '.':
                tail_target += ').as_float())'
            elif print_type_code == '%':
                tail_target += ').as_pct())'
            elif '-' in print_type_code:
                tail_target = '%s).as_string("%s"))' % (tail_target,print_type_code)
            else:
                tail_target += '))'
        return head_target + LeappTranslator.get_target0b(source_fragment) + tail_target  
        
    @staticmethod
    def get_target0b(source_fragment):
        source_fragment = source_fragment.strip()
        is_bool = len(source_fragment) >= 1 and (source_fragment[0] == '@')
        if is_bool:
            source_fragment = source_fragment[1:]
        target = LeappTranslator.get_target1(source_fragment)
        if is_bool:
            target = '(' + target + ').p(True)'
        return target
        
    @staticmethod
    def get_target1(source_fragment):
        source_fragments = []
        current_string_delim = None
        chars = []
        for c in source_fragment:
            if current_string_delim is None:
                if c == '#':
                    break
                if c in STRING_SEPARATORS:
                    current_string_delim = c
                    if len(chars) > 0:
                        source_fragments.append(''.join(chars))
                    chars = []
                chars.append(c)
            elif c == current_string_delim:
                current_string_delim = None
                chars.append(c)
                source_fragments.append(''.join(chars))
                chars = []
            else:
                chars.append(c)
        if len(chars) > 0:
            source_fragments.append(''.join(chars))
        extracted_strings = ['']
        unstringed_fragment_parts = []
        for source_fragment in source_fragments:
            if source_fragment[0] in STRING_SEPARATORS:
                extracted_strings.append(source_fragment)
                unstringed_fragment_parts.append('#')
            else:
                unstringed_fragment_parts.append(source_fragment)
        unstringed_fragment = ''.join(unstringed_fragment_parts)
        unstringed_target_fragment = LeappTranslator.get_target(unstringed_fragment)
        fragment_parts = []
        target_fragment = ''.join(s1+s2 for (s1,s2) in zip(extracted_strings,unstringed_target_fragment.split('#')))
        return target_fragment
    
    @staticmethod
    def convert_fraction(fraction_expression):
        fraction_expression = fraction_expression.strip()
        if '/' in fraction_expression or fraction_expression.endswith('%') \
         or fraction_expression.startswith('0') or fraction_expression.startswith('.'):
            prob_fraction = ProbFraction(fraction_expression)
            return "%s,%s" % (prob_fraction.numerator,prob_fraction.denominator)
        return fraction_expression

    @staticmethod
    def smart_split(fragment,sep):
        level_parentheses = 0
        level_braces = 0
        level_brackets = 0
        parts = []
        chars = []
        chars2 = []
        len_sep = len(sep)
        nb_match_chars = 0
        sep_c = sep[0]
        for c in fragment:
            if c == '(':
                level_parentheses += 1
            elif c == ')':
                level_parentheses -= 1
            elif c == '{':
                level_braces += 1
            elif c == '}':
                level_braces -= 1
            elif c == '[':
                level_brackets += 1
            elif c == ']':
                level_brackets -= 1
            if c == sep_c and level_parentheses+level_braces+level_brackets == 0:
                chars2.append(c)
                nb_match_chars += 1
                if nb_match_chars == len_sep:
                    parts.append(''.join(chars))
                    del chars[:]
                    del chars2[:]
                    nb_match_chars = 0
                sep_c = sep[nb_match_chars]
            else:
                if nb_match_chars > 0:
                    nb_match_chars = 0
                    chars += chars2
                    del chars2[:] 
                    sep_c = sep[0]    
                chars.append(c)
            if level_parentheses<0 or level_braces<0 or level_brackets<0:
                raise LeappTranslator.Error('missing opening delimiter')
        if level_parentheses>0 or level_braces>0 or level_brackets>0:
            raise LeappTranslator.Error('missing closing pairing')
        if len(chars) > 0:
            parts.append(''.join(chars))
        return parts        

    @staticmethod
    def treat_prob_weight_expression(dict_expression):        
        if ':' in dict_expression:
            # dictionary literal: enclose it in brackets
            distribution_items = tuple(LeappTranslator.smart_split(item,':') for item in LeappTranslator.smart_split(dict_expression,','))
            prob_fractions = tuple(ProbFraction(prob_weight_string) for (val_str,prob_weight_string) in distribution_items)
            prob_weights = ProbFraction.get_prob_weights(prob_fractions)
            new_dict_expression = ','.join('(%s,%d)'%(LeappTranslator.get_target(val_str),prob_weight) \
                                 for ((val_str,prob_weight_string),prob_weight) in zip(distribution_items,prob_weights))
            return 'Lea.from_val_freqs(%s)' % new_dict_expression
        # dictionary variable: expand items
        return 'Alea.from_val_freqs_dict_gen(%s)' % dict_expression
        
    @staticmethod
    def treat_cpt_expression(cpt_expression):
        if '->' in cpt_expression:
            # CPT literal
            def f(cond_expr):
                if cond_expr.strip() == '_':
                    return None
                return cond_expr    
            cpt_items = tuple(LeappTranslator.smart_split(item,'->') for item in LeappTranslator.smart_split(cpt_expression,','))
            new_cpt_expression = ','.join('(%s,%s)'%(f(cond_expr),distrib_expr) for (cond_expr,distrib_expr) in cpt_items)
            return new_cpt_expression
        return cpt_expression
        
    @staticmethod
    def get_target(source_fragment):
        target_fragment = source_fragment
        target_fragment = LeappTranslator.parse(target_fragment,'?*','(',')','Lea.cprod(%s)')
        target_fragment = LeappTranslator.parse(target_fragment,'?!' ,'(',')','Lea.build_cpt(*(%s,))',LeappTranslator.treat_cpt_expression)
        target_fragment = LeappTranslator.parse(target_fragment,'!!' ,'(',')','.revised_with_cpt(*(%s,))',LeappTranslator.treat_cpt_expression)
        target_fragment = LeappTranslator.parse(target_fragment,'?' ,'(',')','Lea.from_vals(*(%s))')
        target_fragment = LeappTranslator.parse(target_fragment,'?:','(',')','Lea.bool_prob(%s)',LeappTranslator.convert_fraction)
        target_fragment = LeappTranslator.parse(target_fragment,'?' ,'{','}','%s',LeappTranslator.treat_prob_weight_expression)
        target_fragment = LeappTranslator.parse(target_fragment,'!' ,'','','.given(%s)')
        target_fragment = LeappTranslator.parse(target_fragment,'$_' ,'(',')','.random_draw(%s)')
        target_fragment = LeappTranslator.parse(target_fragment,'$' ,'(',')','.random(%s)')
        target_fragment = LeappTranslator.parse_single(target_fragment,'$','.random()')
        target_fragment = LeappTranslator.parse_single(target_fragment,'.@','.new()')
        target_fragment = LeappTranslator.parse(target_fragment,'@' ,'(',')','.p(%s)')
        target_fragment = LeappTranslator.parse_isolated_at(target_fragment)
        target_fragment = LeappTranslator.parse_single(target_fragment,'@','.p(True)')
        (head,times_args,identifier,tail) = LeappTranslator.parse_identifier(target_fragment,'?')
        if head is not None:
            if identifier is None:
                raise LeappTranslator.Error('missing identifier')
            times_expr = ''
            if times_args is not None:
                times_expr = '.times(%s)'% times_args
            is_func = tail.lstrip().startswith('(')
            if is_func:
                target_fragment = head + LeappTranslator.parse(tail,'','(',')',('Flea.build(%s,(%%s,))'%identifier)+times_expr)
            else:
                target_fragment = head + identifier
                if times_expr:
                    target_fragment += times_expr
                else:
                    target_fragment += '.clone()'
                target_fragment += LeappTranslator.get_target(tail)
        return target_fragment

    @staticmethod
    def parse_isolated_at(source_fragment):
        split_result = source_fragment.split('@',1)
        if len(split_result) == 1:
            return source_fragment
        (head,tail) = split_result
        if len(tail) == 0 or tail[0] not in LITERAL_CHARACTERS:
            return source_fragment
        split_result2 = tail.split(None,1)
        if len(split_result2) == 2:
            (body,tail2) = split_result2
        else:
            body = tail
            tail2 = ''
        return '%s.p(%s) %s' % (head,LeappTranslator.get_target(body),LeappTranslator.get_target(tail2))

    @staticmethod
    def parse_single(source_fragment,prefix_token,target_body):
        split_result = source_fragment.split(prefix_token,1)
        if len(split_result) == 1:
            return source_fragment
        (head,tail) = split_result
        return head + target_body + LeappTranslator.get_target(tail)

    @staticmethod
    def parse(source_fragment,prefix_token,open_token,close_token,target_body_template,treat_func=None):
        special = open_token == ''
        if prefix_token == '':
            (head1,tail1) = ('',source_fragment)
        else:
            split_result1 = source_fragment.split(prefix_token,1)
            if len(split_result1) == 1:
                return source_fragment
            (head1,tail1) = split_result1
            #head1 = head1.rstrip()
        if special:
            if len(tail1) > 0 and tail1[0] == '=':
                # ignore Python inequality operator '!='
                return head1 + '!' + LeappTranslator.parse(tail1,prefix_token,open_token,close_token,target_body_template)
            (head2,tail2) = ('',tail1)
            (open_token,close_token) = ('(',')')
        else:
            split_result2 = tail1.split(open_token,1)
            if len(split_result2) == 1:
                return source_fragment
            (head2,tail2) = split_result2
        if len(head2.lstrip()) > 0:
            return source_fragment
        body_chars = []
        tail_iter = iter(tail2)
        level = 1
        for c in tail_iter:
            if c == open_token:
                level += 1
            elif c == close_token:
                level -= 1
                if level == 0:
                    if special:
                        body_chars.append(close_token)
                    break
            body_chars.append(c)
        if level > 0 and not special:
            raise LeappTranslator.Error("opening '%s' but missing '%s'"%(prefix_token+open_token,close_token))
        body = ''.join(body_chars)
        tail = ''.join(tail_iter)
        if treat_func is not None:
            body = treat_func(body)
        target_body = LeappTranslator.get_target(body)
        return head1 + target_body_template % target_body + LeappTranslator.get_target(tail)
        
    @staticmethod
    def get_identifier(source_fragment):
        identifier_chars = []
        tail_iter = iter(source_fragment)
        tail = ''
        for c in tail_iter:
            if c not in IDENTIFIER_CHARACTERS:
                tail = c
                break
            identifier_chars.append(c)
        identifier = ''.join(identifier_chars)
        tail += ''.join(tail_iter)
        return (identifier,tail)

    @staticmethod
    def parse_identifier(source_fragment,prefix_token):
        (head,times_args,identifier,tail) = (None,None,None,None)
        split_result = source_fragment.split(prefix_token,1)
        if len(split_result) == 2:
            (head,tail1) = split_result
            tail1 = tail1.lstrip()
            if len(tail1) > 0:
                first_char = tail1[0]
                if first_char == '[':
                    split_result = tail1[1:].split(']',1)
                    if len(split_result) == 2:
                        (times_args,tail1) = split_result
                        tail1 = tail1.lstrip()
            if len(tail1) > 0:
                first_char = tail1[0]
                if first_char in string.ascii_letters:
                    (identifier,tail) = LeappTranslator.get_identifier(tail1)
        return (head,times_args,identifier,tail)

