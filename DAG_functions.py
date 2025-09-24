import re
import os
import sys
import math
import pickle
import textwrap
import heapq
from collections import Counter
from itertools import cycle, islice
from typing import (
    Callable, List, Tuple, Sequence, Set,
    Hashable, Dict, Iterable, Union, Pattern, Optional
)

import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_ue_idx_list(lines, number_of_ues):
    ue_idx_list = [[] for _ in range(number_of_ues)]
    
    for i in range(number_of_ues):
        print(f"Scanning UE {i}")
        pattern = re.compile(rf'\bue={i}\b')
        
        for idx, line in tqdm(enumerate(lines), total=len(lines), desc=f"Scanning UE {i}"):
            if pattern.search(line):
                ue_idx_list[i].append(idx)
    
    return ue_idx_list


def classify_ue_layers(
    uex_idx_list: List[int],
    lines: List[str],
    layer_patterns: Dict[str, Pattern]
) -> Dict[str, List[int]]:
    uex_layer_idx = {layer: [] for layer in layer_patterns}

    for idx in tqdm(uex_idx_list, desc="Classifying ue=1 layers"):
        line = lines[idx]
        for layer, pattern in layer_patterns.items():
            if re.search(pattern, line):
                uex_layer_idx[layer].append(idx)

    return uex_layer_idx


def select_uex_sig(idx_list, signals):
    idx_set = set(idx_list)
    comp_sig = [sig for idx, sig in enumerate(signals) if idx in idx_set]
    
    return comp_sig


def slice_by_window(data, window_size=200):
    return [data[i:i+window_size] for i in range(0, len(data), window_size)]


def sliding_windows_step(data, window_size, step):
    return [
        data[i : i + window_size]
        for i in range(0, len(data) - window_size + 1, step)
    ]
    
    
def generate_DAG(
    traces: List[List[Hashable]]
) -> Tuple[
    List[nx.DiGraph],
    List[Dict[Hashable, int]],
    List[Dict[Tuple[Hashable, Hashable], int]]
]:

    graph_list = []
    node_support_list = []
    edge_support_list = []

    for trace in tqdm(traces, desc="Processing traces", total=len(traces)):
        local_graph = nx.DiGraph()
        local_node_support: Dict[Hashable, int] = {}
        local_edge_support: Dict[Tuple[Hashable, Hashable], int] = {}

        for node in trace:
            local_node_support[node] = local_node_support.get(node, 0) + 1

        for u, v in zip(trace, trace[1:]):
            if local_graph.has_node(u) and local_graph.has_node(v):
                if nx.has_path(local_graph, v, u):
                    continue
            local_graph.add_edge(u, v)
            local_edge_support[(u, v)] = local_edge_support.get((u, v), 0) + 1

        graph_list.append(local_graph)
        node_support_list.append(local_node_support)
        edge_support_list.append(local_edge_support)

    return graph_list, node_support_list, edge_support_list


def check_graphs_are_dag(graph_list: List[nx.DiGraph], print_warnings: bool = True) -> Tuple[bool, List[bool]]:

    results = []
    for idx, G in enumerate(graph_list):
        is_dag = nx.is_directed_acyclic_graph(G)
        results.append(is_dag)
        if not is_dag:
            if print_warnings == True:
                print(f"⚠️ Graph #{idx} is not DAG! ")
    all_dag = all(results)
    if all_dag:
        print("✅ All figures are DAG.")
    else:
        print("❌ exist non-DAG graph.")
    return all_dag, results


def filter_graphs(
    dag_flags: List[bool],
    graph_list: List[nx.DiGraph],
    node_support_list: List[Dict[Hashable, int]],
    edge_support_list: List[Dict[Tuple[Hashable, Hashable], int]]
) -> Tuple[
    List[nx.DiGraph],
    List[Dict[Hashable, int]],
    List[Dict[Tuple[Hashable, Hashable], int]]
]:
    filtered_graphs = []
    filtered_node_supports = []
    filtered_edge_supports = []

    for flag, G, ns, es in zip(dag_flags, graph_list, node_support_list, edge_support_list):
        if flag:
            filtered_graphs.append(G)
            filtered_node_supports.append(ns)
            filtered_edge_supports.append(es)

    return filtered_graphs, filtered_node_supports, filtered_edge_supports


def compute_confidences(node_support, edge_support):
    forward_conf = {}
    backward_conf = {}

    for (h, t), es in edge_support.items():
        h_ns = node_support.get(h, 0)
        t_ns = node_support.get(t, 0)

        if h_ns > 0:
            forward_conf[(h, t)] = es / h_ns
        else:
            forward_conf[(h, t)] = 0.0

        if t_ns > 0:
            backward_conf[(h, t)] = es / t_ns
        else:
            backward_conf[(h, t)] = 0.0

    return forward_conf, backward_conf


def compute_all_confidences(
    node_support_list: List[Dict[Hashable, int]],
    edge_support_list: List[Dict[Tuple[Hashable, Hashable], int]],
    compute_confidences_fn
) -> Tuple[List[float], List[float]]:

    forward_confidences: List[float] = []
    backward_confidences: List[float] = []

    for ns, es in tqdm(
        zip(node_support_list, edge_support_list),
        desc="Computing confidences",
        total=len(node_support_list)
    ):
        f_conf, b_conf = compute_confidences_fn(ns, es)
        forward_confidences.append(f_conf)
        backward_confidences.append(b_conf)

    return forward_confidences, backward_confidences


def prune_causality_graph(CG, forward_confidence, backward_confidence, threshold=0.01):

    PG = CG.copy()
    for u, v in list(PG.edges()):
        fwd = forward_confidence.get((u, v), 0)
        bwd = backward_confidence.get((u, v), 0)
        if (fwd + bwd) / 2 < threshold:
            PG.remove_edge(u, v)
    return PG


def prune_graphs(
    graph_list: List[nx.DiGraph],
    forward_confidence: List[float],
    backward_confidence: List[float],
    threshold: float,
    prune_causality_graph
) -> List[nx.DiGraph]:

    pruned_graphs: List[nx.DiGraph] = []

    for G, f_conf, b_conf in tqdm(
        zip(graph_list, forward_confidence, backward_confidence),
        total=len(graph_list),
        desc="Pruning graphs"
    ):
        pruned_G = prune_causality_graph(G, f_conf, b_conf, threshold=threshold)
        pruned_graphs.append(pruned_G)

    return pruned_graphs


def model_selection(PG_list: List[nx.DiGraph], cutoff: int = None) -> List[List]:

    M_total: List[List] = []

    for PG in tqdm(PG_list, desc="Processing graphs", total=len(PG_list)):
        topo = list(nx.topological_sort(PG))
        initial_nodes = [
            n for n in PG.nodes
            if PG.in_degree(n) == 0 and PG.out_degree(n) > 0
        ]
        UM: Set = set(PG.nodes)     
        M: List[Sequence] = []      

        def suffix_table(UM_local: Set) -> dict:
            suf = {v: [] for v in PG.nodes}
            for u in reversed(topo):
                best: List = []
                for v in PG.successors(u):
                    cand = [u] + suf[v]
                    if len(cand) > len(best):
                        best = cand
                if u in UM_local and not best:
                    best = [u]
                if any(x in UM_local for x in best):
                    suf[u] = best
            return suf

        while UM:
            suf = suffix_table(UM)
            progress = False

            for src in initial_nodes:
                path = suf.get(src, [])
                if not path:
                    continue
                # 可选的长度过滤
                if cutoff is not None and len(path) < cutoff:
                    continue
                M.append(path)
                UM.difference_update(path)
                progress = True
                if not UM:
                    break

            if not progress:
                for node in list(UM):
                    if cutoff is None or cutoff <= 1:
                        M.append([node])
                UM.clear()
        M_total.extend(M)

    return M_total


def build_fsa(paths: List[Sequence[str]]
              ) -> Tuple[nx.DiGraph, int]:
    G = nx.DiGraph()
    q0 = 0
    G.add_node(q0)                  
    next_state = 1

    for p in paths:
        curr = q0
        for msg in p:

            nxt = None
            for _, nbr, data in G.out_edges(curr, data=True):
                if data["msg"] == msg:
                    nxt = nbr
                    break
            if nxt is None:             
                nxt = next_state
                next_state += 1
                G.add_edge(curr, nxt, msg=msg)
            curr = nxt
    return G, q0

def model_evaluation(
    traces: Iterable[Sequence[str]],
    model_paths: List[Sequence[str]],
) -> Tuple[float, Set[str], Set[Tuple[int, str, int]]]:

    traces = list(traces)
    

    if traces and isinstance(traces[0], str):
        traces = [traces]


    G, q0 = build_fsa(model_paths)
    

    transitions = {
        u: { d["msg"]: v for _, v, d in G.out_edges(u, data=True) }
        for u in G.nodes()
    }


    unused_edges = {(u, d["msg"], v) for u, v, d in G.edges(data=True)}
    unaccepted = set()
    sum_ar, n_traces = 0.0, 0


    for rho in tqdm(traces, total=len(traces), desc="Evaluating traces"):
        if not rho:
            continue
        n_traces += 1
        accepted = 0
        X = set()  
        for m in rho:

            matched = False
            new_X = set() 
            nxt = transitions[q0].get(m)
            if nxt is not None:
                new_X.add(nxt)
                unused_edges.discard((q0, m, nxt))
                matched = True

            #
            for q in X:
                nxt = transitions[q].get(m)
                if nxt is not None:
                    new_X.add(nxt)
                    unused_edges.discard((q, m, nxt))
                    matched = True
            X |= new_X
            if matched:
                accepted += 1
            else:
                unaccepted.add(m)

        sum_ar += accepted / len(rho)
    ar = (sum_ar / n_traces) if n_traces else 0.0
    return ar, unaccepted, unused_edges

def path_score(
    PG: nx.DiGraph,
    path: Sequence[str],
    forward_confidence: Dict[Tuple[str, str], float],
    backward_confidence: Dict[Tuple[str, str], float]
) -> float:
    if len(path) < 2:
        return 0.0
    fwd_vals = []
    bwd_vals = []
    for u, v in zip(path[:-1], path[1:]):
        fwd_vals.append(forward_confidence.get((u, v), 0.0))
        bwd_vals.append(backward_confidence.get((u, v), 0.0))
    forward_score  = sum(fwd_vals) / len(fwd_vals)
    backward_score = sum(bwd_vals) / len(bwd_vals)
    return (forward_score + backward_score) / (len(path) - 1)

def compute_scores_multi(
    PGs: Iterable[nx.DiGraph],
    forward_confs: Iterable[Dict[Tuple[str, str], float]],
    backward_confs: Iterable[Dict[Tuple[str, str], float]]
) -> List[List[str]]:
    """

    """

    PG_list = list(PGs)

    total = 0
    for PG in tqdm(PG_list, desc="Counting total paths"):
        roots  = [n for n in PG.nodes() if PG.in_degree(n) == 0]
        leaves = [n for n in PG.nodes() if PG.out_degree(n) == 0]
        for r in roots:
            for t in leaves:
                total += sum(1 for _ in nx.all_simple_paths(PG, r, t))
    # total = 0
    # for PG in PGs:
    #     roots  = [n for n in PG.nodes() if PG.in_degree(n) == 0]
    #     leaves = [n for n in PG.nodes() if PG.out_degree(n) == 0]
    #     for r in roots:
    #         for t in leaves:
    #             total += sum(1 for _ in nx.all_simple_paths(PG, r, t))

    # path2scores collect scores for different paths 
    path2scores: Dict[Tuple[str, ...], List[float]] = {}
    pbar = tqdm(total=total, desc="Scoring all PGs")

    # Scoring
    for PG, fwd_conf, bwd_conf in zip(PGs, forward_confs, backward_confs):
        roots  = [n for n in PG.nodes() if PG.in_degree(n) == 0]
        leaves = [n for n in PG.nodes() if PG.out_degree(n) == 0]

        for r in roots:
            for t in leaves:
                for p in nx.all_simple_paths(PG, r, t):
                    pbar.update(1)
                    key = tuple(p)
                    sc = path_score(PG, p, fwd_conf, bwd_conf)
                    path2scores.setdefault(key, []).append(sc)

    pbar.close()

    # Sort
    aggregated: List[Tuple[float, Tuple[str, ...]]] = []
    for path, scores in path2scores.items():
        avg_score = sum(scores) / len(scores)
        aggregated.append((avg_score, path))

    topk = heapq.nlargest(len(aggregated), aggregated, key=lambda x: x[0])
    return [list(path) for _, path in topk]




def model_refinement(
        sorted_paths,
        init_paths: List[Sequence[str]],      # pruned G
        traces:       List[Sequence[str]],    # logs trace      # Pruned causality graph
        accuracy_th:  float = 0.9,  
        max_iteration: int = 30  # max_iteration
) -> Tuple[List[List[str]], float]:
    """
    """
    # Copy!
    M:  List[List[str]] = [list(p) for p in init_paths]
    # sorted_paths = compute_scores(PG, forward_confidence, backward_confidence)
    print("compute_scores finished")

    #
    AR, UM, UE = model_evaluation(traces, M)
    AR = 0
    
    acc_list = []
    
    print("model_evaluation finished, current AR:", AR)
    i = 0

    while i <= max_iteration and AR < accuracy_th and sorted_paths:
        i += 1
        print("Iteration:", i)
        
        
        # Remove paths that contain “unused edge UE” — treat as pruning.
        if UE:
            bad_msgs = {msg for _, msg, _ in UE}
            M = [p for p in M if not any(m in bad_msgs for m in p)]

        # Find the most frequently occurring unaccepted message, maxUnaccepted.
        if not UM:
            break
        cnt = Counter()
        for tr in traces:
            cnt.update(m for m in tr if m in UM)
        max_unaccepted, _ = cnt.most_common(1)[0]

        # Select one candidate path that covers maxUnaccepted and add it to the model.
        chosen = None
        for p in sorted_paths:
            if max_unaccepted in p:
                chosen = p
                break
        if chosen is None:                
            chosen = sorted_paths[0]

        M.append(chosen)
        sorted_paths.remove(chosen)

        # Recompute scores
        AR, UM, UE = model_evaluation(traces, M)
        print("model_evaluation again, current AR:", AR)
        # print("model_evaluation again, current AR:", AR)
        
        acc_list.append(AR)

    return M, AR, acc_list

def visualize_fsa(
    G: nx.DiGraph,
    q0: Optional[Hashable] = None,
    *,
    pos: Optional[Dict[Hashable, Tuple[float, float]]] = None,
    figsize: Tuple[int, int] = (8, 6),
    layout: str = "spring",
    label_edges: bool = True,
    node_size: int = 800,
    font_size: int = 12,
    edge_font_size: int = 10,
    start_node_style: Dict = None,
) -> None:


    # Choose , compute node positions
 
    if pos is None:
        layout = layout.lower()
        if layout == "kamada":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        elif layout == "planar":
            try:
                pos = nx.planar_layout(G)
            except nx.NetworkXException:  # fallback if graph not planar
                pos = nx.spring_layout(G)
        else:  # default: spring layout
            pos = nx.spring_layout(G)


    # Base drawing (nodes & edges)

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_size,
                           node_color="#A0CBE2",
                           edgecolors="black")
    nx.draw_networkx_edges(G, pos,
                           arrowstyle="-|>",
                           arrowsize=10,
                           connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_labels(G, pos,
                            font_size=font_size,
                            font_weight="bold")


    # Edge labels (transition messages)

    if label_edges:
        edge_labels = nx.get_edge_attributes(G, "msg")
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=edge_font_size,
            label_pos=0.5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none")
        )

    # 4 Highlight the start state
    if q0 is not None and q0 in G:
        style = {
            "node_color": "#FFCC66",
            "node_shape": "s",
            "linewidths": 1.5,
            "edgecolors": "black",
            "alpha": 1.0,
        }
        if start_node_style:
            style.update(start_node_style)

        nx.draw_networkx_nodes(G, pos,
                               nodelist=[q0],
                               node_size=node_size * 1.1,
                               **style)

    plt.axis("off")
    plt.tight_layout()
    plt.show()
    

def plot_model_paths(M: List[Sequence], layout: str = "spring",  print_graph: bool = False) -> nx.DiGraph:

    if not M:
        raise ValueError("Empty model, nothing to plot.")

    # build graph
    G = nx.DiGraph()
    for path in M:
        nx.add_path(G, path)
        
    if print_graph == True:
        # pos calculation
        if layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "stair":
            #  k-th path at y = -k
            pos = {}
            for k, path in enumerate(M):
                for i, node in enumerate(path):
                    pos[node] = (i, -k)          # x=i, y=-k
        else:  # spring
            pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(31, 31))
        nx.draw(G, pos,
                with_labels=True,
                node_size=400,
                arrowsize=8,
                linewidths=1,
                font_size=12)
        plt.title("Model M paths")
        plt.axis("off")
        plt.show()
    
    return G


def count_edges_in_graph(G: Union[nx.Graph, nx.DiGraph]) -> int:
    return G.number_of_edges()


def draw_layered_fsm(
    G: nx.DiGraph,
    prog: str = "dot",
    rankdir: str = "TB",
    nodesep: float = 0.3,
    ranksep: float = 0.5,
    figsize=(12, 8),
    node_size=300,
    font_size=8,
    arrowsize=12
):

    G2 = G.copy()
    G2.graph.update({
        "rankdir": rankdir,
        "nodesep": str(nodesep),
        "ranksep": str(ranksep)
    })
    
    # generate position
    try:
        pos = nx.nx_agraph.graphviz_layout(G2, prog=prog)
    except (ImportError, AttributeError):
        pos = nx.nx_pydot.pydot_layout(G2, prog=prog)

    # Plot
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G2, pos,
                           node_size=node_size,
                           node_color="white",
                           edgecolors="black")
    nx.draw_networkx_labels(G2, pos,
                            font_size=font_size)
    nx.draw_networkx_edges(G2, pos,
                           arrowstyle="-|>",
                           arrowsize=arrowsize,
                           connectionstyle="angle3,angleA=0,angleB=90")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
def draw_layered_fsm_note(
    G: nx.DiGraph,
    id_to_log,            # list-of-(id, text) or dict
    prog: str = "dot",
    rankdir: str = "TB",
    nodesep: float = 0.3,
    ranksep: float = 0.5,
    figsize=(12, 8),
    node_size: int = 300,
    font_size: int = 8,
    arrowsize: int = 12,
    wrap_width: int = 30,         
    y_offset_pts: float = -10.0   
):
    if not isinstance(id_to_log, dict):
        id_to_log = dict(id_to_log)
    corrected = {}
    for k, v in id_to_log.items():
        if isinstance(k, str) and k.isdigit():
            corrected[int(k)] = v
        corrected[k] = v
    id_to_log = corrected

    G2 = G.copy()
    G2.graph.update({
        "rankdir": rankdir,
        "nodesep": str(nodesep),
        "ranksep": str(ranksep)
    })

    try:
        pos = nx.nx_agraph.graphviz_layout(G2, prog=prog)
    except (ImportError, AttributeError):
        pos = nx.nx_pydot.pydot_layout(G2, prog=prog)


    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(
        G2, pos,
        node_size=node_size,
        node_color="white",
        edgecolors="black",
        ax=ax
    )
    nx.draw_networkx_labels(G2, pos, font_size=font_size, ax=ax)
    nx.draw_networkx_edges(
        G2, pos,
        arrows=True,
        arrowsize=arrowsize,
        connectionstyle="angle3,angleA=0,angleB=90",
        ax=ax
    )
    
    # connectionstyle="arc3,rad=0.1",

    for n, (x, y) in pos.items():
        if n not in id_to_log:
            continue
        text = id_to_log[n]
        wrapped = textwrap.fill(text, width=wrap_width)
        ax.annotate(
            wrapped,
            xy=(x, y),
            xytext=(0, y_offset_pts),
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize=font_size,
            color='black',
            wrap=True,
            annotation_clip=False
        )

    ax.axis('off')
    plt.tight_layout(pad=1.0)
    plt.show()
    
    
def plot_model_refine_score(accuracy_list):
    plt.figure(figsize=(25, 10)) 
    plt.plot(accuracy_list, marker='s', color='darkblue', label='RLC_Layer_Graph')
    for i, acc in enumerate(accuracy_list):
        plt.text(i, acc, f"{acc:.6f}", ha='center', va='bottom', fontsize=13)
    plt.xlabel('Iterations')           # iterations
    plt.ylabel('Accuracy')        
    plt.title('Model Accuracy')  
    plt.legend()                  
    plt.grid(True)                
    plt.tight_layout()
    plt.show()
    
    
def extract_log_tuples(
    log_lines: List[str],
    G: nx.Graph
) -> List[Tuple[int, str]]:

    graph_ids = set(G.nodes)
    log_tuples: List[Tuple[int, str]] = []

    for line in log_lines:
        if not line.startswith("ID="):
            continue
        try:

            id_str = line.split(":", 1)[0]         
            log_id = int(id_str.split("=", 1)[1])  

            if log_id not in graph_ids:
                continue

 
            cleaned = re.sub(r'^ID=\d+\s*:\s*size=\d+\s*:\s*<:TIMESTAMP:>\s*', '', line)
            first_bracket = re.search(r'\[[^\]]+\]', cleaned)
            cleaned = re.sub(r'\[[^\]]+\]\s*', '', cleaned)
            if first_bracket:
                cleaned = first_bracket.group(0) + ' ' + cleaned

            cleaned = re.sub(r'\bdu=<:NUM:>\s*', '', cleaned)
            cleaned = re.sub(r'\bue=<:NUM:>\s*', '', cleaned)

            cleaned = cleaned.strip()

            log_tuples.append((log_id, cleaned))
        except Exception:
            continue

    return log_tuples


def trim_after_last_period(log_tuples):

    trimmed_tuples = []

    for log_id, line in log_tuples:
        last_period = line.rfind('.')
        if last_period != -1:
            trimmed_line = line[:last_period + 1].strip()
        else:
            trimmed_line = line.strip()

        trimmed_tuples.append((log_id, trimmed_line))

    return trimmed_tuples