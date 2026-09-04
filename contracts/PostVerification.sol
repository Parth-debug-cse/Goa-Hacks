// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract PostVerification {
    struct Record {
        bytes32 contentHash;
        string sourceUrl;
        string metadataURI;
        uint256 timestamp;
        address submitter;
    }

    mapping(uint256 => Record) public records;
    uint256 public nextRecordId;

    event RecordRegistered(uint256 indexed recordId, bytes32 contentHash, string sourceUrl, string metadataURI, address submitter);

    function registerRecord(bytes32 contentHash, string calldata sourceUrl, string calldata metadataURI) external returns (uint256) {
        uint256 recordId = nextRecordId;
        records[recordId] = Record({
            contentHash: contentHash,
            sourceUrl: sourceUrl,
            metadataURI: metadataURI,
            timestamp: block.timestamp,
            submitter: msg.sender
        });
        nextRecordId = recordId + 1;

        emit RecordRegistered(recordId, contentHash, sourceUrl, metadataURI, msg.sender);
        return recordId;
    }

    function verifyRecord(uint256 recordId, bytes32 candidateHash) external view returns (bool) {
        return records[recordId].contentHash == candidateHash;
    }
}
