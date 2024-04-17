import jinja2

# This is a template which will accept a dictionary with the following:
# - title: The title of the page.
# - phase_plots: A list of plotly plots (as divs) to display.
# - video: An optional video to display.
PHASE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        /* Make sure the plots are displayed in a row, and that they don't take up more than a certain width*/
        /* Also make sure the row has a max number of elements = 3 and overflows to the next row */
        .plot-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
        }
        .plot {
            flex: 1; /* Ensures each plot takes equal space */
            padding: 10px; /* Optional: adds space between plots */
            min-width: 500px; /* Optional: sets a minimum width for each plot */
            max-width: 500px; /* Optional: sets a maximum width for each plot */]
        }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <h2>Phase Plots</h2>
    <div class="plot-container">
    {% for plot in phase_plots %}
        <div class="plot">
            {{ plot }}
        </div>
    {% endfor %}
    </div>
    {% if video %}
    <h2>Video</h2>
    <div>
        <video src="{{ video }}" width="640" height="360" controls></video>
    </div>
    {% endif %}
</body>
</html>
"""

### This is a template which will accept a dictionary with the following:
# - title: The title of the page.
# - episode_nums: A list of episode numbers to display, with links to the corresponding pages.
EXPERIMENT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
</head>
<body>
    <h1>{{ title }}</h1>
    <h2>Episodes</h2>
    <ul>
    {% for episode_num in episode_nums %}
        <li><a href="episodes/{{ episode_num }}/index.html">{{ episode_num }}</a></li>
    {% endfor %}
    </ul>
</body>
</html>
"""


def render_episode_page(title, phase_plots, video=None):
    """
    Renders a page with the given title, phase plots, and optional video.
    """
    template = jinja2.Template(PHASE_TEMPLATE)
    return template.render(title=title, phase_plots=phase_plots, video=video)


def render_experiment_page(title, episode_nums):
    """
    Renders a page with the given title and episode numbers.
    """
    template = jinja2.Template(EXPERIMENT_TEMPLATE)
    return template.render(title=title, episode_nums=episode_nums)
