"""Check offline parsing, accounting and copied evidence without importing probes."""
import ast
import gzip
import hashlib
import json
import re
import unittest
from pathlib import Path

from audit_dev import extract_tools, linked_results, option, shell_commands


ROOT=Path(__file__).parent


class ParsingTests(unittest.TestCase):
    def test_heredoc_does_not_become_execution(self):
        commands=shell_commands("cat > scratch/probe.py <<'EOF'\npython tools/sandbox.py --kind dev\nEOF\npython3 agent/optimizer/src/runtime_tools.py gateway-execute --request scratch/dev.json")
        self.assertEqual([name for name,_ in commands],['cat','runtime_tools.py'])

    def test_composite_invocations_remain_separate(self):
        commands=shell_commands('python tools/sandbox.py --kind dev -- python a.py; python tools/sandbox.py --kind profile -- python b.py')
        self.assertEqual(len(commands),2)
        self.assertEqual([option(args,'--kind') for _,args in commands],['dev','profile'])

    def test_quoted_code_is_not_a_cli(self):
        self.assertEqual(shell_commands('python -c "print(\'python tools/sandbox.py --kind dev\')"')[0][0],'python')

    def test_trace_tool_ids_and_results_are_deduplicated(self):
        use={'message':{'content':[{'type':'tool_use','id':'one','name':'Bash','input':{'command':'echo hello'}}]}}
        result={'message':{'content':[{'type':'tool_result','tool_use_id':'one','content':'hello','is_error':False}]}}
        payload='\n'.join(json.dumps(v) for v in [use,use,result,result]).encode()
        tools=extract_tools(payload,False)
        self.assertEqual(len(tools),1);self.assertEqual(len(tools[0]['results']),1)
        wrapped='\n'.join(json.dumps({'event':v}) for v in [use,result]).encode()
        self.assertEqual(extract_tools(wrapped,True)[0]['results'][0]['text'],'hello')

    def test_background_link_requires_matching_task(self):
        tool={'line':2,'results':[{'line':3,'text':'Command running in background with ID: task1. Output is being written to: /tmp/x/tasks/task1.output.'}]}
        other=[{'line':5,'name':'TaskOutput','input':{'task_id':'task1'},'results':[{'line':6,'text':'linked'}]},
               {'line':7,'name':'TaskOutput','input':{'task_id':'task2'},'results':[{'line':8,'text':'unrelated'}]}]
        results,background=linked_results(tool,other)
        self.assertTrue(background);self.assertEqual([r['text'] for r in results][-1],'linked')
        self.assertEqual(len(results),2)


class EvidenceTests(unittest.TestCase):
    def test_accounting_reconciles(self):
        summary=json.loads((ROOT/'data/summary.json').read_text())
        self.assertEqual(summary['scope']['diagnostics']['AKA_sessions'],114)
        self.assertEqual(summary['scope']['diagnostics']['retained_sessions'],120)
        self.assertEqual(sum(summary['retained']['exit_codes'].values()),616)
        self.assertEqual(summary['retained']['unique_jobs'],615)
        self.assertEqual(sum(summary['aka']['requested_kinds'].values()),1630)
        patterns=json.loads((ROOT/'data/usage-patterns.json').read_text())['retained_linked_request_counts']
        self.assertEqual(sum(patterns[k] for k in ('linked_unique_command','ambiguous_command','unlinked')),616)

    def test_per_dsl_command_exit_counts(self):
        rows=json.loads(gzip.decompress((ROOT/'data/retained-dev.json.gz').read_bytes()))
        self.assertEqual(sum(r['exit_code'] not in (None,0) for r in rows),88)
        self.assertEqual(sum(r['exit_code']==0 for r in rows),527)
        self.assertEqual(sum(r['job_status']=='rejected' for r in rows),1)

    def test_copied_evidence_hashes(self):
        for source in (ROOT/'examples').glob('*/provenance.json'):
            meta=json.loads(source.read_text())
            for item in meta['files']:
                if item.get('missing'):
                    continue
                matches=list(source.parent.rglob(item['copy']))
                self.assertTrue(any(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'] for p in matches), (source,item))

    def test_helpers_parse_without_import_or_execution(self):
        for path in (ROOT/'examples').rglob('*.py'):
            ast.parse(path.read_text(),filename=str(path))

    def test_report_links_exist(self):
        for path in (ROOT/'README.md',ROOT/'examples/README.md'):
            for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
                if not target.startswith(('https://','http://','#')):
                    self.assertTrue((path.parent/target.split('#')[0]).exists(),(path,target))


if __name__=='__main__':
    unittest.main()
