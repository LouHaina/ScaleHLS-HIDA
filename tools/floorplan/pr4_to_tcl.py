"""Run the PR's unit-level ILP (proto_floorplan.solve_ilp, same settings as its main())
and write the assignment as a Vivado floorplan Tcl in the same pblock style as the
old LayoutPartition pass output (SLR pblocks, CONTAIN_ROUTING, SNAPPING_MODE),
so only the assignment differs between the two."""
import sys, json, math
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
import proto_floorplan as pf
cpp, out_tcl, out_json = sys.argv[1:4]
K = 3
funcs = pf.parse_cpp(cpp)
units, leaves, parent, edges, areas = pf.build_graph(funcs, 'forward')
tot = {r: sum(areas[u][r] for u in units) for r in ('lut', 'ff', 'dsp', 'bram', 'uram')}
cap = {r: math.ceil(tot[r] * 1.3 / K) for r in tot}
unit_edges = [e for e in edges if e[0] in parent and e[1] in parent and parent[e[0]] != parent[e[1]] or e[3] != 'buffer']
asg, dt, st = pf.solve_ilp(units, unit_edges, areas, K, cap, 23040 // 2, alpha=1)
print('status', st, 'solve_s', round(dt, 2), 'inter-unit edges', len([e for e in unit_edges if e[0] in units and e[1] in units]))
json.dump(dict(assignment=asg, areas={u: areas[u] for u in units}), open(out_json, 'w'), indent=1)
L = ['# ILP floorplan (PR #4 proto_floorplan.py, unit level), emitted for evaluation',
     'catch {delete_pblocks [get_pblocks]}']
for k in range(K):
    us = sorted(u for u in units if asg[u] == k)
    L += [f'# SLR{k}: ' + ' '.join(us),
          f'create_pblock pblock_slr{k}',
          f'resize_pblock pblock_slr{k} -add {{SLR{k}}}',
          f'set_property CONTAIN_ROUTING true [get_pblocks pblock_slr{k}]',
          f'set_property SNAPPING_MODE ON [get_pblocks pblock_slr{k}]',
          'set cells [list]']
    for u in us:
        L += [f'set c [get_cells -quiet -regexp {{^(grp_)?{u}(_[0-9]+)?(_fu_[0-9]+|_U0)$}}]',
              f'if {{[llength $c] != 1}} {{ error "ILP floorplan: {u} matched [llength $c] cells" }}',
              'lappend cells {*}$c']
    L += [f'add_cells_to_pblock pblock_slr{k} $cells',
          f'puts "pblock_slr{k}: [llength $cells] units"']
open(out_tcl, 'w').write('\n'.join(L) + '\n')
for k in range(K):
    us = [u for u in units if asg[u] == k]
    print(f'SLR{k}', len(us), 'units, proxy', {r: sum(areas[u][r] for u in us) for r in ('lut', 'dsp', 'bram')})
