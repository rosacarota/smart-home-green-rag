The eco-metric was validated on 360 generated rule pairs evaluated by three independent LLM judges. 
All three judges evaluated the same examples, allowing a direct inter-judge comparison. 
The judges fully agreed on the final label for 235/360 examples (65.3%), while at least two judges agreed for 353/360 examples (98.1%). 
Only 7/360 examples (1.9%) had no majority label and were therefore marked as ambiguous.

The consensus label distribution was: 253 strong_green, 72 accept, 28 reject, and 7 ambiguous examples. 
Invalidity detection was more judge-dependent: 12 examples were marked invalid by at least one judge, but 0 examples received an invalid majority. 
Therefore, single-judge invalid cases should be manually inspected rather than automatically discarded.

The numerical component of the eco-metric was also stable. 
The average pairwise Spearman correlation was 0.785 for EcoStatic, 0.837 for Delta_energy, and 0.694 for Delta_awareness. 
This suggests that the judges tend to rank generated rules similarly, especially for energy-related improvements.
