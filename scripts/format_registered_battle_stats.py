"""Format the full sealed native outcome table without inference or fitting."""
import argparse,pathlib
from report_registered_combat_comparison import read,sha


def format_stats(report):
    assert report['state']=='complete' and not report['training_performed'] and not report['promotion']
    variants=report['variants']
    assert sum(v['outcomes']['all']['battles'] for v in variants.values())==report['native_cases']
    lines=['# Статистика боёв', '',
        f"Принято {report['native_cases']} native боёв; {report['cuda_cases']} CUDA cases проверены. Ниже исходы первой жизни на matched fixtures; до/после сравниваются одинаковые условия и seeds. Оценочные данные не использованы для обучения.", '',
        '| Вариант | Победы / бои | Смерти | Убийства монстров | Урон нанесён | Урон получен | Ordinary победы | Recovery победы |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,v in variants.items():
        outcomes=v['outcomes'];t=outcomes['all']
        def split(name):
            s=outcomes.get(name)
            return f"{s['goal']}/{s['battles']}" if s else '—'
        fields=[label,f"{t['goal']}/{t['battles']}",t['death'],t['kills'],t['dealt'],t['received'],split('ordinary'),split('recovery')]
        lines.append('| '+' | '.join(map(str,fields))+' |')
    lines+=['', 'Урон — native health damage, не точность выстрелов. Смерть, убийства и выполнение всей цели — разные метрики. Ordinary/recovery имеют разные условия; суммарные результаты не заменяют раздельное сравнение.', '',
        '| Вариант | Смены цели /100 provider кадров | Повороты ≥30° | Стояние при clear enemy | Attack-команды без выбранной цели |',
        '|---|---:|---:|---:|---:|']
    for label,v in variants.items():
        b=(v.get('behavior') or {}).get('totals',{})
        def number(key,percentage=False):
            value=b.get(key)
            return '—' if value is None else f"{value*(100 if percentage else 1):.2f}"+('%' if percentage else '')
        fields=[label,number('target_changes_per_100_frames'),number('large_yaw_command_fraction',True),
            number('stationary_clear_enemy_fraction',True),number('attack_without_target_fraction',True)]
        lines.append('| '+' | '.join(fields)+' |')
    lines+=['','Поворот или стояние сами по себе не доказаны как лишнее действие. Attack-команды не считаются физическими выстрелами или промахами; для rules provider-only behavior metrics отсутствуют и обозначены «—». Смены оружия и вмешательства исполнителя приводятся в отдельных sealed supplementary reports.','',report['scope'],'']
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args();assert not a.out.exists()
    digest=sha(a.report);report=read(a.report)
    for binding in report['inputs']:assert sha(binding['path'])==binding['sha256'],binding['path']
    text=format_stats(report)
    assert sha(a.report)==digest
    a.out.write_text(text,encoding='utf-8')
    print(a.out,flush=True)


if __name__=='__main__':main()
