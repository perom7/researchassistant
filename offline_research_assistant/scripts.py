from __future__ import annotations

from dataclasses import dataclass
from typing import List

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

    host = []
    expert = []

    host.append("Welcome! In simple terms, what is this research trying to achieve?")
    kp = ""
    if keyphrases:
        kp = " Key themes include: " + ", ".join(keyphrases[:5]) + "."
    expert.append(f"At a high level, it tackles {topic}. {analogy}{kp} In practical terms, the paper focuses on: {summary[:220]}...")

    host.append("What approach or method does the paper propose?")
    expert.append(f"Think of it step-by-step: the authors define the problem, choose data or inputs, and then test a solution. Using the analogy, it’s like adjusting how the system behaves to reduce mistakes and improve reliability.")

    host.append("What were the key results or takeaways?")
    expert.append("The key takeaway is that the proposed approach improves performance compared to a baseline. The paper reports results that support the claim, and the main value is clearer decision-making or safer operation.")

    host.append("What are the limitations or risks?")
    expert.append("Like any study, it has constraints: assumptions about data, compute resources, and real-world deployment. In the analogy, it’s like testing a tool in a workshop before using it on a construction site.")

    host.append("If someone wants to build on this work, what’s next?")
    expert.append("Next steps usually include larger datasets, stronger evaluation, and real-world pilots. The paper also suggests future extensions that can make the method more robust and practical.")

    return PodcastScript(host_lines=host[:exchanges], expert_lines=expert[:exchanges])


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
