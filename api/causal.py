import json

def handler(request):
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'ace_ranking': [
                {'name': 'Concept_Retention', 'value': 0.942},
                {'name': 'Time_On_Task', 'value': 0.780},
                {'name': 'Inter_Skill_Dependency', 'value': 0.554},
                {'name': 'Anxiety_Factor', 'value': -0.210}
            ],
            'benchmark_data': [
                {'model': 'NS-CausalKT-v2', 'auc_baseline': 0.824, 'auc_causal': 0.891, 'rmse_delta': -0.062, 'efficiency_gain': 6.1},
                {'model': 'AKT (Baseline)', 'auc_baseline': 0.791, 'auc_causal': None, 'rmse_delta': 0.000, 'efficiency_gain': None},
                {'model': 'DKT (Baseline)', 'auc_baseline': 0.805, 'auc_causal': 0.812, 'rmse_delta': 0.002, 'efficiency_gain': None}
            ]
        })
    }
