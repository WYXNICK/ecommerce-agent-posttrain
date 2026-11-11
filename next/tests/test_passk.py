import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/passk'))
from diagnose import SamplingClient, complete_rows, estimate, load_plan, report, task_values
from shopping_agent.evaluation.summary import _is_strict_success
from shopping_agent.protocol import ROOT, sha256
from shopping_agent.runs import RunStore, evaluate


class PassKTests(unittest.TestCase):
    def test_estimator_matches_exhaustive_subsets(self):
        for c in range(9):
            samples=[1]*c+[0]*(8-c)
            for k in [1,2,4,8]:
                subsets=list(itertools.combinations(samples,k))
                exact=sum(any(s) for s in subsets)/len(subsets)
                self.assertAlmostEqual(task_values([c],8,k)[0],exact)

    def test_task_paired_bootstrap_preserves_zero_difference(self):
        v=task_values([0,1,4,8],8,4)
        indices=np.random.default_rng(42).integers(0,4,(1000,4))
        self.assertEqual(estimate(v-v,indices),{'estimate':0.,'ci95_task_bootstrap':[0.,0.]})

    def test_sampling_is_explicit_and_seed_survives_transport_retry(self):
        c=SamplingClient('http://localhost','student',task_id=17,replicate=0,seed=42,temperature=1.,top_p=1.)
        payload={'messages':[{'role':'user','content':'task'}],'temperature':0,'model':'student'}
        captured=[]
        class Response:
            def __enter__(self):return self
            def __exit__(self,*_):pass
            def read(self):return b'{}'
        def transport(request,timeout):
            captured.append(json.loads(request.data));return Response()
        with patch('diagnose.urlopen',transport):
            c.send('http://localhost',payload,{},180);c.send('http://localhost',payload,{},180)
        self.assertEqual(captured[0],captured[1]);self.assertEqual(captured[0]['temperature'],1.)
        self.assertEqual(payload['temperature'],0)
        other=SamplingClient('http://localhost','student',task_id=17,replicate=1,seed=42,temperature=1.,top_p=1.)
        self.assertNotEqual(c.call_seed(payload['messages']),other.call_seed(payload['messages']))

    def test_resume_retries_only_infrastructure_not_model_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);tasks=[{'task_id':1},{'task_id':2}]
            spec={'candidate_task_ids':[1,2],'execution_fingerprint':'fixed','split':'dev','formal':False}
            def row(task,kind):
                return {'task_id':task,'failure_kind':kind,'status':'failed','execution_fingerprint':'fixed',
                        'replicate':0,'final_reward':0.,'steps':[]}
            called=[]
            with RunStore(directory,spec) as store:
                store.append(row(1,'model'));store.append(row(2,'infrastructure'))
                def episode(task,attempt_index):
                    called.append((task['task_id'],attempt_index));return row(task['task_id'],'model')
                result=evaluate(store,tasks=tasks,episode=episode,workers=1)
            self.assertEqual(called,[(2,1)]);self.assertEqual(result['status'],'complete')
            self.assertEqual(set(complete_rows(directory,[1,2],0,'fixed')),{1,2})
            with (directory/'raw.jsonl').open('a') as f:f.write(json.dumps(row(1,'model'))+'\n')
            with self.assertRaisesRegex(ValueError,'resampled'):complete_rows(directory,[1,2],0,'fixed')

    def test_complete_gold_required(self):
        row={'status':'done','done':True,'terminal_result':{'done':True,'over':True,'reward_detail':{
            'reward_version':'shopsimulator-reward-v3','reward_type':'gold_purchase','reward_valid':True,
            'purchase_success':True,'termination_reason':'gold_purchase'}}}
        self.assertTrue(_is_strict_success(row))
        row['terminal_result']['reward_detail']['reward_valid']=False
        self.assertFalse(_is_strict_success(row))

    def test_frozen_four_models_and_dev_inputs(self):
        plan,models=load_plan(ROOT/'configs/passk.json')
        self.assertEqual([m['name'] for m in models],['s0','s-repair','s0-grpo-450','grpo-500'])
        self.assertEqual(plan['tasks']*plan['samples_per_task']*len(models),9600)
        self.assertEqual(models[0]['base_files'],models[2]['base_files'])
        self.assertEqual(models[1]['base_files'],models[3]['base_files'])

    def test_report_complete_grid_and_paired_differences(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);output=root/'output';(root/'docs/passk').mkdir(parents=True)
            for p in [root/'docs/passk/README.md',root/'docs/project-completion-plan.md']:
                p.write_text('**Pass@k 诊断状态：运行中。**')
            plan={'samples_per_task':8,'sampling':{},'bootstrap_seed':42,'bootstrap_resamples':100,
                  'execution_fingerprint':'fixed','k':[1,2,4,8],'comparisons':[['a','b']]}
            planpath=root/'plan.json';planpath.write_text(json.dumps(plan))
            models=[{'name':name,'greedy_source':'greedy.json','greedy_keys':[]} for name in ['a','b']]
            (root/'greedy.json').write_text(json.dumps({'expected_tasks':2,'completed_tasks':2,'strict_success_task_ids':[1]}))
            for model in models:
                for replica in range(8):
                    d=output/model['name']/f'replicate-{replica:02d}';d.mkdir(parents=True)
                    (d/'run.json').write_text(json.dumps({'plan_sha256':sha256(planpath),'model':model}))
                    (d/'summary.json').write_text(json.dumps({'status':'complete','usage':{'reported_tokens':{'total_tokens':10}}}))
                    rows=[]
                    for task in [1,2]:
                        success=task==1 or (model['name']=='b' and replica==0)
                        rows.append({'task_id':task,'replicate':replica,'execution_fingerprint':'fixed','failure_kind':None,
                            'status':'done','done':True,'steps':[],'final_reward':float(success),
                            'terminal_result':{'done':True,'over':True,'reward_detail':{'reward_version':'shopsimulator-reward-v3',
                            'reward_type':'gold_purchase' if success else 'wrong_purchase','reward_valid':True,
                            'purchase_success':success,'termination_reason':'gold_purchase' if success else 'wrong_purchase'}}})
                    (d/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with patch('diagnose.ROOT',root),patch('diagnose.TaskPool') as pool:
                pool.return_value.tasks.return_value=[{'task_id':1},{'task_id':2}]
                report(SimpleNamespace(output=output,plan=planpath),plan,models)
            result=json.loads((output/'summary.json').read_text())
            self.assertEqual(result['models']['a']['pass_at_k']['8']['estimate'],.5)
            self.assertEqual(result['models']['b']['pass_at_k']['8']['estimate'],1.)
            self.assertEqual(result['comparisons']['a__b']['delta_pass_at_k']['1']['estimate'],1/16)
            self.assertEqual(result['comparisons']['a__b']['observed_coverage_gained'],1)
            self.assertIn('已完成',(root/'docs/passk/README.md').read_text())


if __name__=='__main__':unittest.main()
