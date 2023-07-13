"""Unit tests for the streamlit middleware class"""

from pandasai import PandasAI
from pandasai.llm.fake import FakeLLM
from pandasai.middlewares.flask_charts import FlaskChartsMiddleware
import pytest


class TestFlaskChartsMiddleware:
    """Unit tests for the flask charts middleware class"""

    @pytest.fixture
    def code(self):
        return """
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.title("Example Chart")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
"""

    def test_flask_charts_middleware(self, code):
        """Test the flask charts middleware"""
        middleware = FlaskChartsMiddleware()
        assert (
            middleware(code=code)
            == """import base64
from io import BytesIO
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
fig = Figure()
ax = fig.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 9, 16])
ax.set_title('Example Chart')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
plt.show()
buf = BytesIO()
fig.savefig(buf, format='png')
data = base64.b64encode(buf.getbuffer()).decode('ascii')
print(f'<img src="data:image/png;base64,{data}"/>')"""
        )
        assert middleware.has_run

    def test_flask_chart_middleware_optional_dependency(self, code):
        """Test the flask chart middleware installs the optional dependency"""

        llm = FakeLLM(output=code)
        pandas_ai = PandasAI(
            llm, middlewares=[FlaskChartsMiddleware()], enable_cache=False
        )

        df = []

        pandas_ai(df, "Random chart fake prompt")

        actual_dependencies = pandas_ai._additional_dependencies
        expected_dependencies = [
            {"module": "base64", "name": "base64", "alias": "base64"},
            {"module": "io", "name": "BytesIO", "alias": "BytesIO"},
            {"module": "matplotlib.figure", "name": "Figure", "alias": "Figure"},
            {
                "module": "matplotlib.pyplot",
                "name": "matplotlib.pyplot",
                "alias": "plt",
            },
        ]
        assert (
            actual_dependencies == expected_dependencies
        ), f"Actual: {actual_dependencies}"
