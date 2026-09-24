import json

with open('streamlit_app/run_history/RUN-20260923T135256-04f77b.json', encoding='utf-8') as f:
    rec = json.load(f)

dc = rec['debug_capture']
out = []


def disp(c):
    return c if isinstance(c, str) else (c.get('display_name') or c.get('inn') or str(c))


a11 = dc.get('A11_identity') or {}
out.append(f"=== A11 identity resolved (all candidate identities before grouping): {len(a11)} ===")
for key, v in a11.items():
    comps = [disp(c) for c in (v.get('components') or [])]
    out.append(f"- {key!r} | inn={v.get('inn')!r} | combo={v.get('is_combination')} | components={comps}")

out.append("")
a14 = dc.get('A14_grouping') or {}
out.append(f"=== A14 grouping (candidate groups fed to scope adjudication): {len(a14)} ===")
for key, recs in a14.items():
    out.append(f"GROUP KEY: {key!r} -- {len(recs)} record(s)")
    for r in recs[:2]:
        comp = r.get('comparator') or {}
        out.append(f"    as_stated={comp.get('as_stated')!r} role={comp.get('role')!r} "
                    f"tier={r.get('tier')} member_state={r.get('member_state')!r} "
                    f"source_class={r.get('source_class')!r}")

out.append("")
a12 = dc.get('A12_scope_adjudication') or {}
out.append(f"=== A12 scope adjudication verdicts: {len(a12)} candidates ===")
for key, d in a12.items():
    comp = d.get('comparator') or {}
    by_pop = d.get('adjudication_by_population') or {}
    if by_pop:
        for pid, adj in by_pop.items():
            out.append(f"- {comp.get('as_stated')!r} [{pid}] verdict={adj.get('verdict')} "
                        f"decisive_facet={adj.get('decisive_facet')!r} reason={adj.get('reason')!r}")
    else:
        adj = d.get('adjudication') or {}
        out.append(f"- {comp.get('as_stated')!r} verdict={adj.get('verdict')} "
                    f"decisive_facet={adj.get('decisive_facet')!r} reason={adj.get('reason')!r}")

out.append("")
excl = rec.get('output', {}).get('validation', {}).get('excluded', []) if rec.get('output') else []
out.append(f"=== validation.excluded (final, top-level): {len(excl)} ===")
for e in excl:
    out.append(f"- value={e.get('value')!r} stage={e.get('stage')!r} reason={e.get('reason')!r}")

with open('_analysis_dump.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('done, lines:', len(out))
