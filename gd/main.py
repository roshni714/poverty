from reporting import write_result
import argh
from policy import (
    saturation_policy,
    geographic_policy,
    binary_targeting_policy,
    oracle_policy,
    optimized_policy,
    conditional_optimized_policy,
)


POLICY_METHODS = {
    "saturation": saturation_policy,
    "geographic": geographic_policy,
    "binary": binary_targeting_policy,
    "oracle": oracle_policy,
    "optimized": optimized_policy,
    "conditional_optimized": conditional_optimized_policy,
}


@argh.arg("--uncondtol", default=0.1)
@argh.arg("--pool", default="central")
@argh.arg("--district", default="malawi")
@argh.arg("--policy", default="saturation")
@argh.arg("--save", default="results")
@argh.arg("--numfeatures", default=2)
def main(
    district="mchinji",
    policy="saturation",
    uncondtol=None,
    pool="central",
    save="district_results",
    numfeatures=2,
):

    learning_method = POLICY_METHODS[policy]
    metrics = learning_method(district=district, uncondtol=uncondtol, pool=pool, numfeatures=numfeatures)
    write_result(save + "{}_pool={}.csv".format(district, pool), metrics)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
