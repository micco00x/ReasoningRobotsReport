def get_breakout_lines_formula(lines_symbols):
    # Generate the formula string
    # E.g. for 3 line symbols:
    # "<(!l0 & !l1 & !l2)*;(l0 & !l1 & !l2);(l0 & !l1 & !l2)*;(l0 & l1 & !l2); (l0 & l1 & !l2)*; l0 & l1 & l2>tt"
    pos = list(map(str, lines_symbols))
    neg = list(map(lambda x: "!" + str(x), lines_symbols))

    s = "(%s)*" % " & ".join(neg)
    for idx in range(len(lines_symbols)-1):
        step = " & ".join(pos[:idx + 1]) + " & " + " & ".join(neg[idx + 1:])
        s += ";({0});({0})*".format(step)
    s += ";(%s)" % " & ".join(pos)
    s = "<%s>tt" % s

    return s

class BreakoutCompleteLinesTemporalEvaluator(TemporalEvaluator):
    """Breakout temporal evaluator for delete columns from left to right"""

    def __init__(self, input_space, bricks_cols=3, bricks_rows=3, lines_num=3, gamma=0.99, on_the_fly=False):
        assert lines_num == bricks_cols or lines_num == bricks_rows
        self.line_symbols = [Symbol("l%s" % i) for i in range(lines_num)]
        lines = self.line_symbols

        parser = LDLfParser()


        string_formula = get_breakout_lines_formula(lines)
        print(string_formula)
        f = parser(string_formula)
        reward = 10000

        super().__init__(BreakoutGoalFeatureExtractor(input_space, bricks_cols=bricks_cols, bricks_rows=bricks_rows),
                         set(lines),
                         f,
                         reward,
                         gamma=gamma,
                         on_the_fly=on_the_fly)

    @abstractmethod
    def fromFeaturesToPropositional(self, features, action, *args, **kwargs):
        """map the matrix bricks status to a propositional formula
        first dimension: columns
        second dimension: row
        """
        matrix = features
        lines_status = np.all(matrix == 0.0, axis=kwargs["axis"])
        result = set()
        sorted_symbols = reversed(self.line_symbols) if kwargs["is_reversed"] else self.line_symbols
        for rs, sym in zip(lines_status, sorted_symbols):
            if rs:
                result.add(sym)

        return frozenset(result)

class BreakoutCompleteRowsTemporalEvaluator(BreakoutCompleteLinesTemporalEvaluator):
    """Temporal evaluator for complete rows in order"""

    def __init__(self, input_space, bricks_cols=3, bricks_rows=3, bottom_up=True, gamma=0.99, on_the_fly=False):
        super().__init__(input_space, bricks_cols=bricks_cols, bricks_rows=bricks_rows, lines_num=bricks_rows, gamma=gamma, on_the_fly=on_the_fly)
        self.bottom_up = bottom_up

    def fromFeaturesToPropositional(self, features, action, *args, **kwargs):
        """complete rows from bottom-to-up or top-to-down, depending on self.bottom_up"""
        return super().fromFeaturesToPropositional(features, action, axis=0, is_reversed=self.bottom_up)
