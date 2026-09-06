// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title MatchRegistry — tamper-evident anchors for face-match evidence bundles.
contract MatchRegistry {
    struct Anchor {
        uint64  timestamp;
        address submitter;
        bytes32 subjectCommitment;
        bytes32 evidenceRoot;
    }

    mapping(bytes32 => Anchor) public anchors;

    event MatchAnchored(
        bytes32 indexed recordHash,
        bytes32 indexed subjectCommitment,
        bytes32 evidenceRoot,
        string  cid,
        uint64  timestamp,
        address indexed submitter
    );

    error AlreadyAnchored(bytes32 recordHash, uint64 at);

    function anchor(
        bytes32 recordHash,
        bytes32 subjectCommitment,
        bytes32 evidenceRoot,
        string calldata cid
    ) external {
        Anchor storage existing = anchors[recordHash];
        if (existing.timestamp != 0) revert AlreadyAnchored(recordHash, existing.timestamp);

        anchors[recordHash] = Anchor(
            uint64(block.timestamp), msg.sender, subjectCommitment, evidenceRoot
        );
        emit MatchAnchored(
            recordHash, subjectCommitment, evidenceRoot, cid,
            uint64(block.timestamp), msg.sender
        );
    }

    function isAnchored(bytes32 recordHash) external view returns (bool, uint64) {
        Anchor storage a = anchors[recordHash];
        return (a.timestamp != 0, a.timestamp);
    }
}
