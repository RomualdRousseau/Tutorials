import math
import pyray as pr

import tutorial1.graph as graph
from tutorial1.map import generate_envelope, union_envelopes

EDGE_COLOR = pr.Color(128, 128, 128, 128)


def main():
    pr.set_config_flags(pr.ConfigFlags.FLAG_MSAA_4X_HINT)
    pr.init_window(600, 600, "main")
    
    g = graph.generate_random()
    envelopes = [generate_envelope(edge, 30) for edge in g.edges]
    segments = union_envelopes(envelopes)

    while not pr.window_should_close():
        pr.begin_drawing()
        pr.clear_background(pr.WHITE)  # type: ignore

        for seg in segments:
            seg.draw(1.0, EDGE_COLOR)
            
        # for seg in g.edges:
        #     seg.draw(1.0, EDGE_COLOR)
                
        # for point in points:
        #     point.draw(10.0, EDGE_COLOR)

        pr.end_drawing()

    pr.close_window()


if __name__ == "__main__":
    main()