// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title ProofAnchor
 * @notice Tamper-evident on-chain anchor for biometric verification records.
 * 
 * Invariant Guarantees:
 * - INV-5: NO BIOMETRICS AND NO PII ON-CHAIN.
 *   Only cryptographic digests (recordHash, metadataDigest) and encrypted IPFS CIDs
 *   are stored.
 * - INV-4: BYTE-EXACT REPRODUCIBILITY.
 *   The recordHash is the canonical Keccak-256 / SHA-256 digest of the RFC 8785 JSON record.
 */
contract ProofAnchor {
    struct AnchorRecord {
        bytes32 recordHash;
        string ipfsCid;
        uint256 timestamp;
        address submitter;
        bytes32 metadataDigest;
    }

    // Mapping: recordHash => AnchorRecord
    mapping(bytes32 => AnchorRecord) public records;

    // Ordered list of all anchored record hashes
    bytes32[] public recordHashes;

    event ProofAnchored(
        bytes32 indexed recordHash,
        string ipfsCid,
        uint256 timestamp,
        address indexed submitter
    );

    error RecordAlreadyExists(bytes32 recordHash);
    error InvalidRecordHash();

    /**
     * @notice Anchor a verified record hash and its encrypted IPFS bundle CID.
     * @param recordHash Canonical 32-byte cryptographic digest of the record.
     * @param ipfsCid IPFS CID of the AES-GCM encrypted proof bundle.
     * @param metadataDigest Additional cryptographic commitment or zero.
     */
    function anchorProof(
        bytes32 recordHash,
        string calldata ipfsCid,
        bytes32 metadataDigest
    ) external returns (bool) {
        if (recordHash == bytes32(0)) revert InvalidRecordHash();
        if (records[recordHash].timestamp != 0) revert RecordAlreadyExists(recordHash);

        records[recordHash] = AnchorRecord({
            recordHash: recordHash,
            ipfsCid: ipfsCid,
            timestamp: block.timestamp,
            submitter: msg.sender,
            metadataDigest: metadataDigest
        });
        recordHashes.push(recordHash);

        emit ProofAnchored(recordHash, ipfsCid, block.timestamp, msg.sender);
        return true;
    }

    /**
     * @notice Verify whether a recordHash is anchored on-chain and retrieve its details.
     */
    function verifyProof(bytes32 recordHash) external view returns (
        bool exists,
        string memory ipfsCid,
        uint256 timestamp,
        address submitter,
        bytes32 metadataDigest
    ) {
        AnchorRecord memory rec = records[recordHash];
        if (rec.timestamp == 0) {
            return (false, "", 0, address(0), bytes32(0));
        }
        return (true, rec.ipfsCid, rec.timestamp, rec.submitter, rec.metadataDigest);
    }

    /**
     * @notice Total number of proofs anchored in this contract.
     */
    function totalAnchors() external view returns (uint256) {
        return recordHashes.length;
    }
}
