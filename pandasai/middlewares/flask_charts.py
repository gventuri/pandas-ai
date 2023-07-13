"""
Flask Charts Middleware class

Middleware to handle the charts for Flask apps in PandasAI.
"""

from pandasai.middlewares.base import Middleware
import ast
import logging


class FlaskChartsMiddleware(Middleware):
    """Flask Charts Middleware class"""

    def run(self, code: str) -> str:
        """
        Run the middleware to remove issues with displaying charts in PandasAI.

        Args:
            code (str): Code to be parsed

        Returns:
            str: Parsed code
        """

        if "plt.show()" in code:
            tree = ast.parse(code)
            transformer = TransformMatplotlibAST()
            new_tree = transformer.visit(tree)
            code = ast.unparse(new_tree)

        return code


class TransformMatplotlibAST(ast.NodeTransformer):
    def __init__(self):
        """
        TransformMatplotlibAST class

        Class to transform the AST of the code to remove issues with displaying charts
        in PandasAI.
        """

        self.encountered_savefig = False
        self.savefig_added = False

        self.logger = logging.getLogger("flask_charts")
        self.logger.setLevel(logging.DEBUG)  # Set the minimum logged level to DEBUG

        # Create a file handler
        handler = logging.FileHandler("flask_charts.log")
        handler.setLevel(logging.DEBUG)  # Set the minimum logged level to DEBUG

        # Create a formatter and add it to the handler
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(handler)

        super().__init__()

    def visit_Expr(self, node):
        """
        Visit the expression node in the AST and transform it if necessary.

        Args:
            node (Expr): Expression node in the AST

        Returns:
            Expr: Transformed expression node in the AST
        """

        call = node.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
            attr = call.func
            if isinstance(attr.value, ast.Name) and attr.value.id == "plt":
                function_args = ", ".join(ast.unparse(arg) for arg in call.args)
                # consider plt function calls with more than one argument
                if len(call.args) > 1:
                    new_code = (
                        "\nfig = Figure()\nax = fig.subplots()\nax.{}({})".format(
                            attr.attr, function_args
                        )
                    )
                    return ast.parse(new_code).body
                elif attr.attr == "title":
                    title_args = ast.unparse(call.args[0]) if call.args else ""
                    return ast.parse("ax.set_title({})".format(title_args)).body
                elif attr.attr == "xlabel":
                    xlabel_args = ast.unparse(call.args[0]) if call.args else ""
                    return ast.parse("ax.set_xlabel({})".format(xlabel_args)).body
                elif attr.attr == "ylabel":
                    ylabel_args = ast.unparse(call.args[0]) if call.args else ""
                    return ast.parse("ax.set_ylabel({})".format(ylabel_args)).body
                elif attr.attr == "savefig":
                    self.encountered_savefig = True
                elif attr.attr == "show" or attr.attr == "close":
                    if not self.encountered_savefig:
                        self.encountered_savefig = True
        return node

    def visit_Module(self, node):
        """
        Visit the module node in the AST and transform it if necessary.

        Args:
            node (Module): Module node in the AST

        Returns:
            Module: Transformed module node in the AST
        """

        import_stmts = ast.parse(
            "import base64\nfrom io import BytesIO\nfrom matplotlib.figure "
            "import Figure",
        ).body
        transformed_body = [self.visit(n) for n in node.body]
        if self.encountered_savefig and not self.savefig_added:
            transformed_body.extend(
                ast.parse(
                    "\nbuf = BytesIO()\nfig.savefig(buf, format='png')\n"
                    "data = base64.b64encode(buf.getbuffer()).decode('ascii')\n"
                    "print(f'<img src=\"data:image/png;base64,{data}\"/>')"
                ).body
            )
            self.savefig_added = True
        return ast.Module(
            body=import_stmts + transformed_body, type_ignores=node.type_ignores
        )
