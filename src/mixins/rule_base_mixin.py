import pandas as pd
from tabulate import tabulate


class RuleBaseMixin:

    def get_lt_rules(self) -> pd.DataFrame:
        """Get the linguistic terms of the rules.

        Returns:
            pd.DataFrame: A DataFrame containing the linguistic terms of the rules.
        """
        if not hasattr(self.rule_base_, "linguistic_rules_"):
            raise ValueError(
                "The model has not been fitted yet. Please call fit() before get_lt_rules()."
            )

        table_data = []
        for i, rule in enumerate(
            self.rule_base_.linguistic_rules_
        ):  # Show first 3 rules
            # print(f"\nRule {i+1} (Cluster {rule['cluster']}):")

            # Build IF part
            if_parts = []
            for ant in rule["antecedents"]:
                if_parts.append(f"{ant['feature_name']} is {ant['linguistic_term']}")
            if_part = " AND ".join(if_parts)

            # Build THEN part
            then_parts = []
            for cons in rule["consequents"]:
                then_parts.append(f"{cons['target_name']} is {cons['linguistic_term']}")
            then_part = " AND ".join(then_parts)

            # print(f"  IF {if_part} THEN {then_part}")
            table_data.append(
                [f"Rule {i+1}", f"Cluster {rule['cluster_index']}", if_part, then_part]
            )

        headers = ["Rule", "Cluster", "IF (Antecedents)", "THEN (Consequents)"]
        print(
            tabulate(
                table_data,
                headers=headers,
                tablefmt="grid",
                maxcolwidths=[None, None, 40, None],
            )
        )

        rules_df = pd.DataFrame(table_data, columns=headers)

        return rules_df
