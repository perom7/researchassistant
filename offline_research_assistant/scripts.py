from __future__ import annotations

from dataclasses import dataclass
from typing import List

import random

from .text_utils import split_sentences

from .keywords import extract_keywords


def build_analogy(topic: str) -> str:
    t = topic.lower()
    if any(k in t for k in ["network", "internet", "routing", "traffic"]):
        return f"{topic} is like a city highway system because data moves in lanes and intersections, and congestion affects speed." 
    if any(k in t for k in ["security", "encryption", "privacy", "cryptography"]):
        return f"{topic} is like a locked mailbox because only the right key can open it, even if others can see the box." 
    if any(k in t for k in ["learning", "model", "training", "classifier"]):
        return f"{topic} is like coaching a team because repeated practice and feedback improves performance over time." 
    if "quantum" in t:
        return f"{topic} is like using fragile seals on a package because any tampering leaves a clear trace." 
    return f"{topic} is like a well-organized toolbox because each part has a job and together they solve a bigger problem." 


@dataclass(frozen=True)
class PodcastScript:
    host_lines: List[str]
    expert_lines: List[str]


def generate_podcast_script(
    summary: str,
    keywords: List[str] | None = None,
    keyphrases: List[str] | None = None,
    topic_hint: str | None = None,
    exchanges: int = 5,
) -> PodcastScript:
    if keywords is None:
        keywords = [k for k, _ in extract_keywords(summary, top_k=10)]

    topic = topic_hint or (keywords[0] if keywords else "this research")
    analogy = build_analogy(topic)

    exchanges = max(3, int(exchanges))

    sents = split_sentences(summary)
    if not sents:
        sents = [summary.strip()] if summary.strip() else []

    # Light heuristic buckets based on cue words.
    buckets = {
        "problem": [],
        "method": [],
        "results": [],
        "limitations": [],
        "impact": [],
    }

    for s in sents:
        sl = s.lower()
        if any(k in sl for k in ["we propose", "we present", "we introduce", "approach", "method", "framework", "algorithm", "model"]):
            buckets["method"].append(s)
        elif any(k in sl for k in ["results", "outperform", "improve", "accuracy", "f1", "auc", "performance", "gain", "increase", "decrease"]):
            buckets["results"].append(s)
        elif any(k in sl for k in ["limitation", "limitations", "however", "but", "assumption", "constraints", "future work", "we leave", "threats"]):
            buckets["limitations"].append(s)
        elif any(k in sl for k in ["impact", "practical", "real-world", "application", "deployment", "benefit", "matters"]):
            buckets["impact"].append(s)
        else:
            buckets["problem"].append(s)

    # Build a conversation plan: start with problem, then method/results, then limitations/impact.
    plan: List[tuple[str, str]] = []

    def add_pair(q: str, a: str) -> None:
        plan.append((q.strip(), a.strip()))

    # Intro
    kp = ""
    if keyphrases:
        kp = " Key themes include: " + ", ".join(keyphrases[:6]) + "."
    add_pair(
        "Alright, give us the big picture—what problem is this paper trying to solve?",
        f"At a high level, it tackles {topic}. {analogy}{kp} In plain terms, the paper is focused on: { (buckets['problem'][0] if buckets['problem'] else (sents[0] if sents else 'a concrete technical problem')) }",
    )

    # Method
    if buckets["method"]:
        add_pair(
            "So what’s the core idea or method they use to go after that problem?",
            f"The central approach is: {buckets['method'][0]} The key is how they structure the steps—inputs, processing, and evaluation—so the solution is testable rather than just a claim.",
        )
    else:
        add_pair(
            "How do they actually approach it? What’s the method in one pass?",
            "They lay out the task, define the inputs and assumptions, then propose a concrete procedure and evaluate it against a baseline. The value is in turning the problem into something measurable and repeatable.",
        )

    # Results
    if buckets["results"]:
        add_pair(
            "And what do the results say—what’s the strongest takeaway?",
            f"The key takeaway is captured here: {buckets['results'][0]} In practice, that means the proposed method is more reliable than the comparison they used, at least under the paper’s evaluation setup.",
        )
    else:
        add_pair(
            "What should listeners remember as the main takeaway?",
            "The headline is whether the proposed approach improves the target outcome compared to a baseline, and why. Even when the gains are modest, the reasoning behind them often tells you when the method will or won’t generalize.",
        )

    # Fill with summary-grounded discussion points.
    fillers = []
    for s in (buckets["method"][1:3] + buckets["results"][1:3] + buckets["problem"][1:3]):
        fillers.append(s)

    random.seed(0)
    random.shuffle(fillers)

    question_starters = [
        "Can you unpack that a bit?",
        "What does that mean in practice?",
        "Why is that detail important?",
        "How should someone interpret that?",
        "If you had to explain that to a new student, how would you say it?",
    ]

    for s in fillers:
        add_pair(
            random.choice(question_starters),
            f"Sure. {s} Put simply, it’s one of the levers that affects performance and reliability—so it’s not just an implementation detail, it changes what the method can handle.",
        )

    # Limitations + impact + next steps
    if buckets["limitations"]:
        add_pair(
            "No paper is perfect—what limitations or caveats should we keep in mind?",
            f"One clear caveat is: {buckets['limitations'][0]} Practically, this tells you what assumptions might break in real deployment and what would need more testing.",
        )
    else:
        add_pair(
            "What are the main limitations?",
            "Typical limitations are dataset size, evaluation scope, compute constraints, and assumptions that simplify reality. The important part is identifying which of those could change the conclusion.",
        )

    if buckets["impact"]:
        add_pair(
            "Where does this matter in the real world?",
            f"In terms of impact: {buckets['impact'][0]} The practical value is that it can guide design decisions or reduce risk when the same kind of problem shows up in production.",
        )
    else:
        add_pair(
            "Why should someone outside this niche care?",
            "Because it’s about making a system more accurate, robust, or understandable—and those themes show up everywhere from software reliability to applied science.",
        )

    add_pair(
        "If someone wanted to extend this work, what would you do next?",
        "I’d push on broader evaluation: more datasets, stronger baselines, ablations that isolate what actually drives the improvement, and then a small real-world pilot to see what breaks.",
    )

    # If we still need more exchanges, add keyword-driven Q/A.
    kwords = [k for k in (keywords or []) if k][:10]
    for k in kwords:
        add_pair(
            f"Quick one: how does {k} fit into the story here?",
            f"It’s one of the key concepts. In this context, {k} helps explain either the problem framing or the mechanism the method relies on. The useful question is: what would change if {k} were different?",
        )
        if len(plan) >= exchanges:
            break

    host_lines = [q for q, _ in plan][:exchanges]
    expert_lines = [a for _, a in plan][:exchanges]

    # Ensure exact length.
    while len(host_lines) < exchanges:
        host_lines.append("Before we wrap up, what’s one detail people often miss when reading papers like this?")
        expert_lines.append("Look for the assumptions hidden in the setup—what data is excluded, what conditions are idealized, and what baseline is chosen. Those choices often explain the results as much as the method does.")

    return PodcastScript(host_lines=host_lines[:exchanges], expert_lines=expert_lines[:exchanges])


def podcast_script_to_text(script: PodcastScript, *, include_speaker_labels: bool = True) -> str:
    lines: List[str] = []
    for h, e in zip(script.host_lines, script.expert_lines):
        if include_speaker_labels:
            lines.append(f"Host: {h}")
            lines.append(f"Expert: {e}")
        else:
            lines.append(h)
            lines.append(e)
        lines.append("")
    return "\n".join(lines).strip()


def generate_video_script(summary: str, kind: str = "reel") -> str:
    kind = (kind or "reel").lower()
    if kind == "reel":
        return (
            "Hook: Here’s the surprising problem this paper tackles.\n"
            f"Problem: {summary[:200]}...\n"
            "Simple idea: The authors propose a practical method to reduce errors and improve reliability.\n"
            "Impact: This matters because it can improve real-world outcomes and reduce risk.\n"
            "Call to action: If you want the full details, check the paper and the slides."
        )

    return (
        "Intro: Today we break down a research paper in plain language.\n"
        f"Context: {summary[:250]}...\n"
        "Method: The paper designs a solution, tests it, and compares to alternatives.\n"
        "Findings: Results support improved performance and clearer decision-making.\n"
        "Conclusion: The approach is promising, with limitations and future improvements identified."
    )
