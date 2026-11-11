"""Retention/correction controls, leakage boundaries and exact LoRA continuation."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

NEXT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(NEXT/'scripts/consolidation'))
from workflow import make_arms, paired_changes, select_candidate, settings_and_policy, verify_initial_policy
from train_sft import initialize_adapter
from shopping_agent.tasks import TaskPool
from sft_data import selected_data


def candidate(ids, reward=.7, step=1):
    return {'step':step,'summary':{'status':'complete','completed_tasks':300,'expected_tasks':300,
        'unresolved_infrastructure_tasks':[], 'strict_successes':len(ids),'strict_success_task_ids':ids,
        'mean_final_reward':reward,'average_steps':9.}}


class ConsolidationTests(unittest.TestCase):
    def test_control_shares_anchors_changes_only_one_block_and_is_order_independent(self):
        rows = lambda start,n:[{'task_id':i} for i in range(start,start+n)]
        settings = {'minimum_pairs':2,'repair_target':2,'seed':42}
        repairs,successes,teachers=rows(1,2),rows(10,5),rows(20,7)
        arms=make_arms(repairs,successes,teachers,set(),settings)
        self.assertEqual(arms,make_arms(repairs[::-1],successes[::-1],teachers[::-1],set(),settings))
        ids={k:{r['task_id'] for r in v} for k,v in arms.items()}
        self.assertEqual(len(ids['repair']),6);self.assertEqual(len(ids['replay']),6)
        self.assertEqual(ids['repair']-ids['replay'],{1,2})
        self.assertEqual(len(ids['repair'] & ids['replay']),4)
        with self.assertRaisesRegex(ValueError,'protected'):
            make_arms(repairs,successes,teachers,{1},settings)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            make_arms(repairs,repairs,teachers,set(),settings)
        with self.assertRaisesRegex(ValueError,'insufficient'):
            make_arms(repairs[:1],successes,teachers,set(),settings)

    def test_baseline_kept_for_reward_only_gain_and_incomplete_evaluation_rejected(self):
        initial=candidate([1,2,3],step=0)
        reward_only=candidate([1,2,3],reward=.9)
        sacrificed_reward=candidate([1,2,3,4],reward=.6)
        self.assertIs(select_candidate(initial,[reward_only,sacrificed_reward]),initial)
        improved=candidate([1,2,3,4],reward=.71)
        self.assertIs(select_candidate(initial,[improved]),improved)
        incomplete=deepcopy(improved);incomplete['summary']['completed_tasks']=299
        with self.assertRaises(ValueError):select_candidate(initial,[incomplete])

    def test_paired_retention_breaks_equal_success_ties(self):
        initial=candidate([1,2,3],step=0)
        churn=candidate([1,2,4,5],reward=.9)
        retain=candidate([1,2,3,4],reward=.71)
        changes=paired_changes(churn['summary'],initial['summary'])
        self.assertEqual((changes['gained'],changes['lost'],changes['retained']),(2,1,2))
        self.assertIs(select_candidate(initial,[churn,retain]),retain)

    def test_frozen_pool_is_disjoint_from_teacher_and_all_holdouts(self):
        settings,policy=settings_and_policy();pool=TaskPool()
        ids={r['task_id'] for r in pool.tasks(settings['mining_split'])}
        self.assertEqual(len(ids),2000)
        self.assertFalse(ids & pool.protected_ids('train'))
        rows,_,_=selected_data()
        self.assertFalse(ids & {r['task_id'] for role in rows.values() for r in role})
        self.assertEqual(policy['name'],'grpo-500')

    def test_wrong_initial_adapter_is_rejected_before_running(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'base').mkdir();(root/'base/weights').write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'identity'):
                verify_initial_policy(root,{'base':'base','base_files':{'weights':'not-the-hash'}})

    def test_adapter_continuation_preserves_logits_and_updates_only_lora(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(42)
        base=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
                                        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1))
        untouched=deepcopy(base)
        trained=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,lora_dropout=0.,target_modules=['q_proj','v_proj']))
        with torch.no_grad():
            for n,p in trained.named_parameters():
                if 'lora_B' in n:p.normal_(std=.1)
        ids=torch.tensor([[1,2,3,4]])
        trained.eval();expected=trained(ids).logits.detach()
        with tempfile.TemporaryDirectory() as name:
            trained.save_pretrained(name)
            continued=initialize_adapter(untouched,{},Path(name))
            continued.eval();torch.testing.assert_close(continued(ids).logits,expected,rtol=0,atol=0)
            before={n:p.detach().clone() for n,p in continued.named_parameters()}
            optimizer=torch.optim.AdamW((p for p in continued.parameters() if p.requires_grad),lr=1e-3)
            continued(ids,labels=ids).loss.backward();optimizer.step()
            changed=[n for n,p in continued.named_parameters() if not torch.equal(p,before[n])]
            self.assertTrue(changed);self.assertTrue(all('lora_' in n for n in changed))
            self.assertEqual(continued.peft_config['default'].lora_dropout,0.)


if __name__=='__main__':unittest.main()
