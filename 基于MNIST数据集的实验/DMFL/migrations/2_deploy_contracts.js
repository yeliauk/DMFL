var DMFL = artifacts.require("./DMFL.sol");

module.exports = function(deployer) {
  deployer.deploy(DMFL);
}

// 这是原代码中两个智能合约的部署
// var Crowdsource = artifacts.require("./Crowdsource.sol");
// var Consortium = artifacts.require("./Consortium.sol");

// module.exports = async function(deployer) {
//   let crowdsource = deployer.deploy(Crowdsource);
//   let consortium = deployer.deploy(Consortium);
//   await Promise.all([crowdsource, consortium]);
// };

