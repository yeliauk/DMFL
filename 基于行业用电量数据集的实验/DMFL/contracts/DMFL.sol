pragma solidity >=0.4.21 <0.7.0;

contract DMFL {
    /// @notice Address of contract creator, who evaluates updates
    address public evaluator;

    /// @notice IPFS CID of genesis model
    bytes32 public genesis;

    /// @dev The timestamp when the genesis model was upload
    uint256 internal genesisTimestamp;

    /// @dev Duration of each training round in seconds
    uint256 internal roundDuration;

    /// @dev Number of updates before training round automatically ends. (If 0, always wait the full roundDuration)
    uint256 internal maxNumUpdates;

    /// @dev Total number of seconds skipped when training rounds finish early
    uint256 internal timeSkipped;

    /// @dev The IPFS CIDs of model updates in each round
    mapping(uint256 => bytes32[]) internal updatesInRound;

    /// @dev The round to which each model update belongs
    mapping(bytes32 => uint256) internal updateRound;

    /// @dev The IPFS CIDs of model updates made by each address
    mapping(address => bytes32[]) internal updatesFromAddress;

    /// @dev Whether or not each model update has been evaluated
    mapping(bytes32 => bool) internal tokensAssigned;

    /// @dev The contributivity score for each model update, if evaluated
    mapping(bytes32 => uint256) internal tokens;



    /// @dev Registration fee in the contract per round
    mapping(uint => uint[]) internal registFee;

    /// @dev reward in the contract per round
    mapping(uint => uint[]) internal reward;

    /// @dev model line in the contract per round
    mapping(uint => uint[]) internal modelLine;

    /// @dev Number of contract types
    uint numIcontractType;



    /// @dev Record the types of contracts signed by the participants in each round
    mapping(address => uint[]) internal signContractRecord;

    /// @dev Record the fees paid by the parties after each round of contract selection
    mapping(address => uint[]) internal feePaidRecord;




    /// @dev trainer msg.sender of each trainer
    mapping(uint => address) internal trainerMsgSender;




    /// @dev Record the relevant content of the incentive mechanism contract
    function icontractRecord(uint _icontractType, uint _registFee, uint _reward, uint _modelLine) external {
        registFee[_icontractType].push(_registFee);
        reward[_icontractType].push(_reward);
        modelLine[_icontractType].push(_modelLine);
    }

    /// @dev Record the number of contract types
    function numIcontractTypeRecord(uint _typeNum) external {
        numIcontractType = _typeNum;
    }



    /// @dev Get the fee for a given contract type
    function getAFee(uint _icontractType, uint _round) external view returns (uint fee){
        fee = registFee[_icontractType][_round-1];
    }



    /// @dev Participants sign contracts in each round
    function signContract(uint _icType, uint _feePaid) external {
        signContractRecord[msg.sender].push(_icType);
        feePaidRecord[msg.sender].push(_feePaid);
    }


    /// @dev Record msg.sender of each trainer
    function trainerMsgSenderRecord(uint _index) external {
        trainerMsgSender[_index] = msg.sender;
    }

    /// @dev Get msg.sender of a trainer
    function getTrainerMsgSender(uint _index) external view returns (address addr) {
        addr = trainerMsgSender[_index];
    }

    /// @dev Get the type of contract signed by a participant in a certain round
    function getTrainerICType(address _addr, uint _round) external view returns (uint ictype) {
        ictype = signContractRecord[_addr][_round-1];
    }

    /// @dev Get the modelLine corresponding to a contract
    function getICModelLine(uint _type, uint _round) external view returns (uint ml) {
        ml = modelLine[_type][_round-1];
    }

    /// @dev Get the trainer model hash for the given address
    function getTrainerHash(address _addr, uint _round) external view returns (bytes32 hash) {
        hash = updatesFromAddress[_addr][_round-1];
    }


    function getICReward(uint _type, uint _round) external view returns (uint r) {
        r = reward[_type][_round-1];
    }



    /// @notice Constructor. The address that deploys the contract is set as the evaluator.
    constructor() public {
        evaluator = msg.sender;
    }

    modifier evaluatorOnly() {
        require(msg.sender == evaluator, "Not the registered evaluator");
        _;
    }

    /// @return round The index of the current training round.
    function currentRound() public view returns (uint256 round) {
        uint256 timeElapsed = timeSkipped + now - genesisTimestamp;
        round = 1 + (timeElapsed / roundDuration);
    }

    /// @return remaining The number of seconds remaining in the current training round.
    function secondsRemaining() public view returns (uint256 remaining) {
        uint256 timeElapsed = timeSkipped + now - genesisTimestamp;
        remaining = roundDuration - (timeElapsed % roundDuration);
    }

    /// @return The CID's of updates in the given training round.
    function updates(uint256 _round) external view returns (bytes32[] memory) {
        return updatesInRound[_round];
    }

    /// @return count Token count of the given address up to and including the given round.
    function countTokens(address _address, uint256 _round)
        external
        view
        returns (uint256 count)
    {
        bytes32[] memory updates = updatesFromAddress[_address];
        for (uint256 i = 0; i < updates.length; i++) {
            bytes32 update = updates[i];
            if (updateRound[update] <= _round) {
                count += tokens[updates[i]];
            }
        }
    }

    /// @return count Total number of tokens up to and including the given round.
    function countTotalTokens(uint256 _round) external view returns (uint256 count) {
        for (uint256 i = 1; i <= currentRound(); i++) {
            bytes32[] memory updates = updatesInRound[i];
            for (uint256 j = 0; j < updates.length; j++) {
                bytes32 update = updates[j];
                if (updateRound[update] <= _round){
                    count += tokens[updates[j]];
                }
            }
        }
    }


    /// @return Get the number of rewards a participant has received this round
    function countCurrentToken(address _address, uint256 _round)
        external
        view
        returns (uint256 count)
    {
        bytes32 model_hash = updatesFromAddress[_address][_round-1];
        count = tokens[model_hash];
    }

    /// @return Get the total number of rewards received by all participants in this round
    function countCurrentTotalTokens(uint256 _round) external view returns (uint256 count) {
        bytes32[] memory current_round_updates = updatesInRound[_round];
        for (uint256 j = 0; j < current_round_updates.length; j++) {
            bytes32 current_round_update = current_round_updates[j];
            if (updateRound[current_round_update] == _round){
                count += tokens[current_round_updates[j]];
            }
        }
    }





    /// @return Whether the given address made a contribution in the given round.
    function madeContribution(address _address, uint256 _round)
        public
        view
        returns (bool)
    {
        for (uint256 i = 0; i < updatesFromAddress[_address].length; i++) {
            bytes32 update = updatesFromAddress[_address][i];
            if (updateRound[update] == _round) {
                return true;
            }
        }
        return false;
    }

    /// @notice Sets a new evaluator.
    function setEvaluator(address _newEvaluator) external evaluatorOnly() {
        evaluator = _newEvaluator;
    }

    /// @notice Starts training by setting the genesis model. Can only be called once.
    /// @param _cid The CID of the genesis model
    /// @param _roundDuration Number of seconds per training round
    /// @param _maxNumUpdates Number of updates per round before training round automatically ends. (If 0, always wait the full roundDuration)
    /// @dev Does not reset the training process! Deploy a new contract instead.
    function setGenesis(
        bytes32 _cid,
        uint256 _roundDuration,
        uint256 _maxNumUpdates
    ) external evaluatorOnly() {
        require(genesis == 0, "Genesis has already been set");
        genesis = _cid;
        genesisTimestamp = now;
        roundDuration = _roundDuration;
        maxNumUpdates = _maxNumUpdates;
    }

    /// @notice Records a training contribution in the current round.
    function addModelUpdate(bytes32 _cid, uint256 _round) external {
        require(_round > 0, "Cannot add an update for the genesis round");
        require(
            _round >= currentRound(),
            "Cannot add an update for a past round"
        );
        require(
            _round <= currentRound(),
            "Cannot add an update for a future round"
        );
        require(
            !madeContribution(msg.sender, _round),
            "Already added an update for this round"
        );

        updatesInRound[_round].push(_cid);
        updatesFromAddress[msg.sender].push(_cid);
        updateRound[_cid] = _round;

        if (
            maxNumUpdates > 0 && updatesInRound[_round].length >= maxNumUpdates
        ) {
            // Skip to the end of training round
            timeSkipped += secondsRemaining();
        }
    }

    /// @notice Assigns a token count to an update.
    /// @param _cid The update being rewarded
    /// @param _numTokens The number of tokens to award; should be based on marginal value contribution
    function setTokens(bytes32 _cid, uint256 _numTokens)
        external
        evaluatorOnly()
    {
        require(!tokensAssigned[_cid], "Update has already been rewarded");
        tokens[_cid] = _numTokens;
        tokensAssigned[_cid] = true;
    }
}
