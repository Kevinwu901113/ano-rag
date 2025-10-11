from typing import List, Dict, Any, Optional, Tuple

from utils.nq_normalize import normalize_text, is_yes_no_question, YES, NO


def _score_yes_no_with_entailment(
    question: str,
    evidence_sents: List[str],
    verifier,
) -> Optional[str]:
    """
    用蕴含打分器(verify_shell.Verifier)来判断 yes/no。
    简化实现：把问句改写成肯定/否定两种命题，让 verifier 打分，选高者。
    verifier.infer(premise, hypothesis) -> score(float)
    """
    if not evidence_sents:
        return None
    premise = " ".join(evidence_sents[:3])  # 取前几句证据
    # 极简改写：直接以问句为命题的正反两种
    hyp_yes = question
    hyp_no = f"NOT: {question}"
    try:
        s_yes = verifier.infer(premise, hyp_yes)
        s_no = verifier.infer(premise, hyp_no)
    except Exception:
        return None
    if s_yes <= 0 and s_no <= 0:
        return None
    return YES if s_yes >= s_no else NO


def _should_abstain(
    conf_scores: Dict[str, float],
    coverage: float,
    entailment_ok: bool,
    yesno_mode: bool,
) -> bool:
    """
    极简无答案判定：多信号联合；你可以把这几个阈值写到 config。
    """
    # 置信度阈值
    ans_conf = conf_scores.get("answer_conf", 0.0)
    support_conf = conf_scores.get("support_conf", 0.0)
    if yesno_mode:
        # 是非问更苛刻：需要证据一致性好
        if not entailment_ok and support_conf < 0.5:
            return True
    else:
        # 普通问句：答案置信低、coverage很低、或证据矛盾则放弃
        if ans_conf < 0.25 or coverage < 0.2:
            return True
    return False


def decide_nq_answer(
    example: Dict[str, Any],
    efsa_candidate: str,
    support_sents: List[str],
    conf_scores: Dict[str, float],
    verifier,
) -> Tuple[str, bool]:
    """
    返回: (predicted_answer, predicted_answerable)
    - efsa_candidate: 你的 EFSA/span pipeline 给出的最优短答案（原始文本）
    - support_sents: 证据句（若为空会降级）
    - conf_scores: {'answer_conf':..., 'support_conf':..., 'entailment':...}
    - verifier: 你的 verify_shell 里的 verifier 对象
    """
    q = example.get("question", "")
    yesno_mode = is_yes_no_question(q)

    # 1) yes/no
    if yesno_mode:
        label = _score_yes_no_with_entailment(q, support_sents, verifier)
        if label is None:
            # 无法判断则视为无答案
            return ("", False)
        # yes/no 输出严格小写
        return (label, True)

    # 2) 普通问句：无答案判定
    entailment_ok = conf_scores.get("entailment", 0.0) >= 0.0  # 你可以改为更严格阈值
    if _should_abstain(
        conf_scores,
        conf_scores.get("coverage", 0.0),
        entailment_ok,
        yesno_mode=False,
    ):
        return ("", False)

    # 3) 文本短答案：规范化输出
    ans = normalize_text(efsa_candidate or "")
    # 过短/空的回退
    if not ans:
        return ("", False)
    return (ans, True)
