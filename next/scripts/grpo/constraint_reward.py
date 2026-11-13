"""Non-compensatory purchase constraints from the frozen task and comparator evidence."""


def purchase_violations(terminal):
    """Use task-contract requirements, not a new natural-language or LLM judge."""
    detail = terminal.get('reward_detail') or {}
    goal = terminal.get('goal') or {}
    sources = goal.get('feature_sources') or {}
    dimensions = detail.get('evidence', {}).get('preference_scoring', {}).get('dimensions', {})
    violations = []
    for dimension, source, comparator in (
            ('brand', 'instruction_explicit_alias', 'explicit_brand_alias_text_v1'),
            ('model', 'instruction_target_token_intersection', 'model_token_boundary')):
        if sources.get(dimension) != source:
            continue
        for result in dimensions.get(dimension, {}).get('results', []):
            required = result['required']
            values = required if isinstance(required, list) else [required]
            if (result.get('comparator') == comparator and result.get('status') == 'fail'
                    and result.get('verifiable') is True and result.get('actual')
                    and values and all(v in goal.get('expected_'+dimension, []) for v in values)):
                violations.append({'constraint': dimension, 'source': source,
                    'required': required, 'actual': result['actual'], 'reason': 'wrong_required_'+dimension})
    for result in dimensions.get('key_options', {}).get('results', []):
        if (result.get('comparator') != 'option_key_value_v1'
                or result.get('status') != 'fail' or result.get('verifiable') is not True):
            continue
        for wrong in result['evidence']['wrong_values']:
            requirement = result['required'][wrong['axis']]
            if requirement.get('source') != 'instruction.instruction_options':
                continue
            violations.append({
                'constraint': 'key_options', 'axis': wrong['axis'], 'source': requirement['source'],
                'source_axis': requirement['source_axis'],
                'required': wrong['required'], 'selected': wrong['selected'],
                'reason': 'omitted_required_option' if wrong['selected'] is None
                          else 'wrong_required_option',
            })
        for unavailable in result['evidence']['unavailable_values']:
            requirement = result['required'][unavailable['axis']]
            if requirement.get('source') == 'instruction.instruction_options':
                violations.append({'constraint': 'key_options', 'axis': unavailable['axis'],
                    'source': requirement['source'], 'source_axis': requirement['source_axis'],
                    'required': unavailable['value'], 'actual': result['actual'],
                    'reason': 'purchased_product_lacks_required_option'})
    if sources.get('core_functions') == 'instruction.attributes':
        instruction = goal.get('instruction_text', '').casefold()
        for result in dimensions.get('core_functions', {}).get('results', []):
            if (result.get('comparator') != 'structured_attribute_with_polarity_fallback'
                    or result.get('status') != 'fail' or result.get('verifiable') is not True):
                continue
            # Missing keywords are not proof of absence. Only explicit contradictory evidence counts.
            for value in result['evidence']['negated']:
                if value in goal.get('expected_core_functions', []) and value.casefold() in instruction:
                    violations.append({'constraint': 'core_functions', 'source': sources['core_functions'],
                        'required': value, 'actual': result['actual'], 'reason': 'required_function_negated'})
    return violations


def constrained_return(trajectory, original_reward, settings):
    """Keep raw Reward v3 intact; a proven violation is sufficient to reject a purchase."""
    terminal = trajectory.get('terminal_result') or {}
    detail = terminal.get('reward_detail') or {}
    violations = []
    if (trajectory['failure_kind'] != 'infrastructure' and trajectory['done']
            and trajectory['status'] == 'done' and terminal.get('done') is True and terminal.get('over') is True
            and detail.get('reward_version') == 'shopsimulator-reward-v3'
            and detail.get('reward_type') in ('partial_alternative_purchase', 'reward_unverifiable')):
        violations = purchase_violations(terminal)
    reward = settings['wrong_purchase_reward'] if violations else original_reward
    return reward, {'version': settings['version'],
                    'decision': 'penalized' if violations else
                                'filtered' if original_reward is None else 'unchanged',
                    'restored_model_failure': bool(violations) and original_reward is None,
                    'violations': violations}
