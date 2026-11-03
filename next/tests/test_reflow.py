"""Replay equivalence and the teacher-only supervision boundary."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/reflow'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data import ReplayClient, repair_point, example_for, repaired_row
from shopping_agent.protocol import load_config, StudentTokenizer, validate_execution
from shopping_agent.runtime import run_episode
from shopping_agent.collection.sft import build_sft_row
from shopping_agent.training.sft.dataset import build_supervised_example
from sft_data import selected_data
from test_execution import Environment, Client, Tokenizer, message


class ReflowTests(unittest.TestCase):
    def trajectory(self):
        return run_episode({'task_id':1},client=Client([
            message('search_products',{'query':'product'},'c1'),message('buy_now',call_id='c2')]),
            config=load_config(),tokenizer=Tokenizer(),base_url='unused',fingerprint='test',env_factory=Environment)

    def test_replay_reexecutes_prefix_and_teacher_sees_same_observations(self):
        source=self.trajectory()
        i=next(i for i,m in enumerate(source['messages']) if any(c['id']=='c2' for c in m.get('tool_calls',[])))
        teacher=Client([message('buy_now',call_id='teacher')])
        client=ReplayClient(source,{'message_index':i,'reason':'test'},teacher)
        result=run_episode({'task_id':1},client=client,config=load_config(),tokenizer=Tokenizer(),
                           base_url='unused',fingerprint='test',env_factory=Environment)
        self.assertTrue(result['done']);self.assertTrue(client.verified)
        self.assertEqual(teacher.seen[0],source['messages'][:i])
        self.assertEqual(len(result['steps']),2)
        row,reasons=repaired_row(result,client)
        self.assertFalse(reasons);self.assertEqual(row['supervise_from_message'],i)

    def test_replay_refuses_changed_observation_before_teacher_call(self):
        source=self.trajectory();teacher=Client([])
        client=ReplayClient(source,{'message_index':4},teacher)
        changed=deepcopy(source['messages'][:2]);changed[1]['content']='different task'
        with self.assertRaisesRegex(ValueError,'observation_mismatch'):client.complete(changed,[],512)
        self.assertEqual(teacher.seen,[])

    def test_only_visible_failure_evidence_defines_takeover(self):
        row=self.trajectory();self.assertIsNone(repair_point(row))
        row['terminal_result']['reward_detail']['reward_type']='wrong_purchase'
        row['failure_kind']='model'
        self.assertEqual(repair_point(row)['reason'],'before_failed_purchase')
        row['failure_kind']='unverifiable';self.assertIsNone(repair_point(row))
        row['failure_kind']='infrastructure';self.assertIsNone(repair_point(row))

    def test_real_template_masks_entire_student_prefix_without_changing_input(self):
        rows,_,_=selected_data();row=deepcopy(rows['train'][0])
        tokenizer=StudentTokenizer.load(load_config())
        indices=[i for i,m in enumerate(row['messages']) if m['role']=='assistant']
        start=indices[len(indices)//2];row['supervise_from_message']=start
        full=build_supervised_example(row['messages'],row['tools'],tokenizer,max_length=24576)
        masked=example_for(row,tokenizer,24576)
        prefix=build_supervised_example(row['messages'][:start],row['tools'],tokenizer,max_length=24576)
        self.assertEqual(full['input_ids'],masked['input_ids'])
        self.assertTrue(all(t==-100 for t in masked['labels'][:len(prefix['input_ids'])]))
        self.assertGreater(sum(t!=-100 for t in masked['labels']),0)
        self.assertLess(sum(t!=-100 for t in masked['labels']),sum(t!=-100 for t in full['labels']))
        last=max(i for i,t in enumerate(full['labels']) if t!=-100)
        self.assertEqual(masked['labels'][last],full['labels'][last])

    def test_frozen_execution_contract_is_unchanged(self):
        self.assertEqual(validate_execution(),'e66a82e73e641dc14460fec97c46038f75ee77866e8cbec6ed382fc9e5422227')

if __name__=='__main__':unittest.main()
