# How to convert a private key from WIF/HEX format to legacy and SegWit addresses to speed up the process, we will use the [Bloom filter algorithm](https://keyhunters.ru/how-to-convert-a-private-key-from-wif-hex-format-to-legacy-and-segwit-addresses-to-speed-up-the-process-we-will-use-the-bloom-filter-algorithm/)

<!-- wp:image {"lightbox":{"enabled":false},"id":2521,"sizeSlug":"large","linkDestination":"custom","align":"center"} -->
<figure class="wp-block-image aligncenter size-large"><a href="https://cryptodeeptech.ru/discrete-logarithm" target="_blank" rel=" noreferrer noopener"><img src="https://keyhunters.ru/wp-content/uploads/2025/03/visual-selection-4-1024x666.png" alt="" class="wp-image-2521"/></a></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><a href="https://polynonce.ru/author/polynonce/"></a></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Below is an example Python script that converts a Bitcoin private key to an address, converts a private key from WIF/HEX format to legacy and SegWit addresses. To speed things up, we will use the Bloom filter algorithm, although its direct application in this context may not be practical, since Bloom filters are typically used to quickly filter data, not to convert keys. However, we can use it to optimize address verification.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Required libraries</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To run the script you will need the following libraries:</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><code>bitcoinaddress</code></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><code>pycryptodome</code> <em>(for working with cryptographic functions)</em></li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>Install them using pip:</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">bash<code>pip install bitcoinaddress pycryptodome
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Script</h2>
<!-- /wp:heading -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>from bitcoinaddress import Wallet
from Crypto.Hash import SHA256
import mmh3

def private_key_to_address(private_key_wif):
    """Преобразует приватный ключ WIF в адрес."""
    wallet = Wallet(private_key_wif)
    return wallet.address

def private_key_to_segwit_address(private_key_wif):
    """Преобразует приватный ключ WIF в SegWit-адрес."""
    wallet = Wallet(private_key_wif)
    return wallet.p2sh_segwit_address

def bloom_filter(addresses):
    """Простой пример использования Bloom filter для фильтрации адресов."""
    <em># Инициализация Bloom filter</em>
    size = 1000000
    hash_functions = 7
    bit_array = [0] * size

    def add(address):
        for seed in range(hash_functions):
            result = mmh3.hash(address, seed) % size
            bit_array[result] = 1

    def lookup(address):
        for seed in range(hash_functions):
            result = mmh3.hash(address, seed) % size
            if bit_array[result] == 0:
                return False
        return True

    <em># Добавляем адреса в Bloom filter</em>
    for address in addresses:
        add(address)

    return lookup

<em># Пример использования</em>
private_key_wif = "5HqrbgkWPqBy6dvCE7FoUiMuiCfFPRdtRsyi6NuCM2np8qBZxq5"
address = private_key_to_address(private_key_wif)
segwit_address = private_key_to_segwit_address(private_key_wif)

print(f"Адрес: {address}")
print(f"SegWit-адрес: {segwit_address}")

<em># Пример использования Bloom filter</em>
addresses = [address, segwit_address]
lookup_func = bloom_filter(addresses)

<em># Проверка адреса</em>
print(f"Адрес {address} найден: {lookup_func(address)}")
print(f"Адрес {segwit_address} найден: {lookup_func(segwit_address)}")
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Notes</h2>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Bloom Filter</strong> : This example uses Bloom filter to demonstrate how you can quickly check for addresses in a dataset. However, it is not practical for actually checking balances or working with transactions.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Key Conversion</strong> : The script converts the private key to a legacy and SegWit address.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Libraries</strong> : The library is used to work with Bitcoin addresses and keys <code>bitcoinaddress</code>. For Bloom filter, it is used <code>mmh3</code>for hashing.</li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>This script demonstrates basic operations with Bitcoin private keys and their conversion to various address formats. More complex operations, such as checking balances or sending transactions, may require additional libraries and APIs.</p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Which Python libraries are best to use for working with Bitcoin</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To work with Bitcoin in Python, you can use the following libraries:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>bitcoinlib</strong> : This is a powerful library that offers a wide range of tools for working with the Bitcoin blockchain. It allows you to create different types of wallets, interact with the blockchain, create and validate transactions, and generate Bitcoin addresses in different formats <a href="https://bitnovosti.io/2025/01/18/python-bitcoin-crypto/" target="_blank" rel="noreferrer noopener">1 </a><a href="https://sky.pro/media/kak-ispolzovat-python-dlya-raboty-s-blokchejn/" target="_blank" rel="noreferrer noopener">5</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>python-bitcoin-utils</strong> : This library is useful for working with Bitcoin transactions and provides various useful functions for analyzing and manipulating data <a href="https://qna.habr.com/q/839255" target="_blank" rel="noreferrer noopener">3</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>block-io</strong> : This library can be used to integrate Bitcoin payments into applications. It provides a simple API for working with transactions and wallets <a href="https://habr.com/ru/articles/525638/" target="_blank" rel="noreferrer noopener">4 </a><a href="https://vc.ru/dev/1616346-vnedryaem-oplatu-btc-kuda-ugodno-python" target="_blank" rel="noreferrer noopener">6</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>blockchain</strong> : This library allows you to interact with the Bitcoin blockchain, retrieve transaction and address information, and create new transactions <a href="https://habr.com/ru/articles/525638/" target="_blank" rel="noreferrer noopener">4 </a><a href="https://vc.ru/dev/1616346-vnedryaem-oplatu-btc-kuda-ugodno-python" target="_blank" rel="noreferrer noopener">6</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>CCXT</strong> : Although primarily used for trading cryptocurrency exchanges, CCXT can also be useful for obtaining market data and creating trading strategies related to Bitcoin <a href="https://bitnovosti.io/2025/01/18/python-bitcoin-crypto/" target="_blank" rel="noreferrer noopener">1</a> .</li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>These libraries allow developers to create a variety of Bitcoin-related applications, from simple transaction analysis scripts to complex trading bots and payment systems.</p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:heading -->
<h2 class="wp-block-heading">How to Convert WIF Private Key to HEX and Back</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Converting a Bitcoin private key between WIF and HEX formats can be done using Python. Below is a code example that shows how to do this:</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Convert HEX to WIF</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To convert a key from HEX to WIF format, you can use a library&nbsp;<code>base58</code>to encode and add a checksum.</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>import hashlib
import base58

def hex_to_wif(hex_key):
    <em># Добавляем префикс '80' для приватного ключа в формате WIF</em>
    hex_key = '80' + hex_key
    
    <em># Вычисляем контрольную сумму</em>
    checksum = hashlib.sha256(hashlib.sha256(bytes.fromhex(hex_key)).digest()).digest()[:4]
    
    <em># Добавляем контрольную сумму к ключу</em>
    hex_key += checksum.hex()
    
    <em># Кодирование Base58</em>
    wif_key = base58.b58encode(bytes.fromhex(hex_key)).decode('utf-8')
    
    return wif_key

<em># Пример использования</em>
hex_private_key = "4BBWF74CQ25A2A00409D0B24EC0418E9A41F9B5B86216A183E0E9731F4589DC6"
wif_private_key = hex_to_wif(hex_private_key)
print(f"WIF Private Key: {wif_private_key}")
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Convert WIF to HEX</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To convert back from WIF to HEX, you can use Base58 decoding and prefix and checksum removal.</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>import base58
import hashlib

def wif_to_hex(wif_key):
    <em># Декодирование Base58</em>
    decoded_key = base58.b58decode(wif_key)
    
    <em># Удаление контрольной суммы и префикса '80'</em>
    hex_key = decoded_key[1:-4].hex()
    
    return hex_key

<em># Пример использования</em>
wif_private_key = "5JPuWYZx922hXi46Lg2RJPrLfqGmkGS9YegMNgiNvx8cJa6kPK8"
hex_private_key = wif_to_hex(wif_private_key)
print(f"HEX Private Key: {hex_private_key}")
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p>These functions allow you to convert private keys between WIF and HEX formats. To work with these functions, you need to have the library installed&nbsp;<code>base58</code>, which can be installed using pip:</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">bash<code>pip install base58
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">What are the tools to check the balance of a Bitcoin address</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To check the balance of a Bitcoin address, you can use the following tools:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Blockchain Explorers</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Blockchair</strong> : Supports multiple blockchains including Bitcoin.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Blockchain.com</strong> : Allows you to view transactions and balances.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Coin.Space</strong> : A simple tool to check the balance of Bitcoin addresses.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Online services</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Crypto.ru</strong> : Offers a tool for quickly checking Bitcoin balance by wallet address.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>MATBEA SWAP</strong> : Allows you to instantly check the balance of all wallet addresses.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>API and software tools</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Bitcoin Core (bitcoind)</strong> : Use the command <code>getbalance</code>to check balance via RPC API.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Python libraries</strong> : Use libraries like <code>bitcoinlib</code>or <code>pycryptodome</code>to interact with the blockchain and check balances programmatically.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Mobile applications</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Key Hunter</strong> : Allows you to check the balance of Bitcoin addresses using a private key.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>These tools make it easy to check the balance of Bitcoin addresses using public blockchain data.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">What errors can occur when working with Bitcoin private keys</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>The following errors and vulnerabilities may occur when working with Bitcoin private keys:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Incorrect ECDSA implementation</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>An incorrect implementation of the ECDSA algorithm can lead to the leakage of private keys. For example, a vulnerability <code>DeserializeSignature</code>allowed attackers to create fake signatures that could be accepted as correct <a href="https://habr.com/ru/articles/817237/" target="_blank" rel="noreferrer noopener">1</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Weak random number generators</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>If the random number generator does not generate truly random data, private keys may be predictable and vulnerable to brute force. This may lead to theft of funds <a href="https://ru.tradingview.com/news/forklog:3031939c867b8:0/" target="_blank" rel="noreferrer noopener">2 </a><a href="https://xakep.ru/2018/04/17/not-so-random/" target="_blank" rel="noreferrer noopener">3</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Weak Brainwallet</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Using memorable phrases to generate private keys can make them vulnerable to guessing, as such phrases are often not random enough <a href="https://ru.tradingview.com/news/forklog:3031939c867b8:0/" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Random Vulnerability</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>If an application uses the same nonce for different transactions, attackers can extract private keys from the signatures <a href="https://ru.tradingview.com/news/forklog:3031939c867b8:0/" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Incorrect storage of keys</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Private keys must be kept secret. If they fall into the hands of attackers, funds can be stolen <a href="https://baltija.eu/2020/07/09/shest-veshei-kotorye-bitkoinery-doljny-znat-o-privatnyh-kluchah/" target="_blank" rel="noreferrer noopener">5</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Hash collisions</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>While it is theoretically possible for different private keys to have the same hash (e.g. ripemd160), in practice this is extremely unlikely and does not pose a significant threat <a href="https://github.com/svtrostov/oclexplorer/issues/6" target="_blank" rel="noreferrer noopener">8</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Runtime attacks</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Some attacks may rely on analysis of the execution time of operations, which may allow attackers to obtain information about private keys <a href="https://habr.com/ru/articles/817237/" target="_blank" rel="noreferrer noopener">1</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Vulnerabilities in software wallets</strong> :<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Bugs in wallet applications can lead to private keys being leaked or misused <a href="https://ru.tradingview.com/news/forklog:3031939c867b8:0/" target="_blank" rel="noreferrer noopener">2 </a><a href="https://xakep.ru/2018/04/17/not-so-random/" target="_blank" rel="noreferrer noopener">3</a> .</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">Citations:</h3>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/817237/">https://habr.com/ru/articles/817237/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.tradingview.com/news/forklog:3031939c867b8:0/">https://ru.tradingview.com/news/forklog:3031939c867b8:0/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://xakep.ru/2018/04/17/not-so-random/">https://xakep.ru/2018/04/17/not-so-random/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://tangem.com/ru/blog/post/entropy/">https://tangem.com/ru/blog/post/entropy/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://baltija.eu/2020/07/09/shest-veshei-kotorye-bitkoinery-doljny-znat-o-privatnyh-kluchah/">https://baltija.eu/2020/07/09/shest-veshei-kotorye-bitkoinery-doljny-znat-o-privatnyh-kluchah/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.securitylab.ru/blog/personal/%20Informacionnaya_bezopasnost_v_detalyah/343072.php">https://www.securitylab.ru/blog/personal/%20Informacionnaya_bezopasnost_v_detalyah/343072.php</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.ixbt.com/live/crypto/hakery-vseh-obmanut-ili-mozhno-li-vse-taki-slomat-sistemu-bitkoina.html">https://www.ixbt.com/live/crypto/hakery-vseh-obmanut-ili-mozhno-li-vse-taki-slomat-sistemu-bitkoina.html</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://github.com/svtrostov/oclexplorer/issues/6">https://github.com/svtrostov/oclexplorer/issues/6</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://crypto.ru/proverit-bitcoin-koshelek/">https://crypto.ru/proverit-bitcoin-koshelek/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://blog.bitbanker.org/ru/kak-posmotret-balans-lyubogo-kriptokoshelka/">https://blog.bitbanker.org/ru/kak-posmotret-balans-lyubogo-kriptokoshelka/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.beincrypto.com/top-5-besplatnyh-platform-onchein-analiza/">https://ru.beincrypto.com/top-5-besplatnyh-platform-onchein-analiza/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://qna.habr.com/q/107877">https://qna.habr.com/q/107877</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://crypto.ru/blockchain-address/">https://crypto.ru/blockchain-address/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://coinsutra.com/ru/crypto-airdrop-checker-tools/">https://coinsutra.com/ru/crypto-airdrop-checker-tools/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://play.google.com/store/apps/details?id=io.github.keyhunter">https://play.google.com/store/apps/details?id=io.github.keyhunter</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://coin.space/ru/bitcoin-address-check/">https://coin.space/ru/bitcoin-address-check/</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/773412/">https://habr.com/ru/articles/773412/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.xn--90abjnskvm1g.xn--p1ai/BitcoinPrivateKey_to_BitcoinAllKeys/index.html">https://www.xn--90abjnskvm1g.xn--p1ai/BitcoinPrivateKey_to_BitcoinAllKeys/index.html</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/682220/">https://habr.com/ru/articles/682220/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://waymorr.ru/news/blog/chto-takoe-privatnyij-klyuch-bitkoin-koshelka">https://waymorr.ru/news/blog/chto-takoe-privatnyij-klyuch-bitkoin-koshelka</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://forum.bits.media/index.php?%2Ftopic%2F178950-%D1%83%D1%82%D0%B8%D0%BB%D0%B8%D1%82%D0%B0-%D0%B2%D0%BE%D1%81%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D1%8F-%D0%BF%D0%BE%D0%B2%D1%80%D0%B5%D0%B6%D0%B4%D1%91%D0%BD%D0%BD%D1%8B%D1%85-%D0%BA%D0%BB%D1%8E%D1%87%D0%B5%D0%B9-%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%82%D0%B0-wif%2F">https://forum.bits.media/index.php?%2Ftopic%2F178950-%D1%83%D1%82%D0%B8%D0%BB%D0%B8%D 1%82%D0%B0-%D0%B2%D0%BE%D1%81%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BB%D0%B5%D0%BD%D 0%B8%D1%8F-%D0%BF%D0%BE%D0%B2%D1%80%D0%B5%D0%B6%D0%B4%D1%91%D0%BD%D0%BD%D1%8B%D1%85-% D0%BA%D0%BB%D1%8E%D1%87%D0%B5%D0%B9-%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%82%D0%B0-wif%2F</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="http://bitcoin-zarabotat.ru/kak-konvertirovat-kljuchi-hex-v-wif-kak-matematicheskaja-zadacha/">http://bitcoin-zarabotat.ru/kak-konvertirovat-kljuchi-hex-v-wif-kak-matematicheskaja-zadacha/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://gist.github.com/Jun-Wang-2018/3105e29e0d61ecf88530c092199371a7">https://gist.github.com/Jun-Wang-2018/3105e29e0d61ecf88530c092199371a7</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://secretscan.org/PrivateKeyWif">https://secretscan.org/PrivateKeyWif</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://bitnovosti.io/2025/01/18/python-bitcoin-crypto/">https://bitnovosti.io/2025/01/18/python-bitcoin-crypto/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://proglib.io/p/kak-python-primenyaetsya-v-blokcheyn-2021-03-19">https://proglib.io/p/kak-python-primenyaetsya-v-blokcheyn-2021-03-19</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://qna.habr.com/q/839255">https://qna.habr.com/q/839255</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/525638/">https://habr.com/ru/articles/525638/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://sky.pro/media/kak-ispolzovat-python-dlya-raboty-s-blokchejn/">https://sky.pro/media/kak-ispolzovat-python-dlya-raboty-s-blokchejn/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://vc.ru/dev/1616346-vnedryaem-oplatu-btc-kuda-ugodno-python">https://vc.ru/dev/1616346-vnedryaem-oplatu-btc-kuda-ugodno-python</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://dzen.ru/a/ZBXLlMLW0G807WSq">https://dzen.ru/a/ZBXLlMLW0G807WSq</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://sky.pro/media/kak-ispolzovat-python-dlya-raboty-s-kriptovalyutami/">https://sky.pro/media/kak-ispolzovat-python-dlya-raboty-s-kriptovalyutami/</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://python-forum.io/archive/index.php/thread-16797.html">https://python-forum.io/archive/index.php/thread-16797.html</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/525638/">https://habr.com/ru/articles/525638/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.stackoverflow.com/questions/1475317/bitcoin-%D0%B0%D0%B4%D1%80%D0%B5%D1%81%D0%B0-%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%82">https://ru.stackoverflow.com/questions/1475317/bitcoin-%D0%B0%D0%B4%D1%80%D0%B5%D1%81%D0%B0-%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%82</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://miningclub.info/threads/keyhunter-py-poisk-privatkey-bitcion-do-2012-na-otformatirovannyx-diskax.31532/">https://miningclub.info/threads/keyhunter-py-poisk-privatkey-bitcion-do-2012-na-otformatirovannyx-diskax.31532/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://pikabu.ru/story/deshifrovanie_paroley_dlya_dostupa_k_poteryannyim_bitkoin_koshelkam_s_pomoshchyu_metoda_gaussjacobi_i_protsessa_mashinnogo_obuchenie_bitcoinchatgpt_11298411">https://pikabu.ru/story/deshifrovanie_paroley_dlya_dostupa_k_poteryannyim_bitkoin_koshelkam_s_pomoshchyu_metoda_gaussjacobi_i_protsessa_mashinnogo_obuchenie_bitcoinchatgpt_11298411</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/674812/">https://habr.com/ru/articles/674812/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.programmersforum.ru/showthread.php?t=327290">https://www.programmersforum.ru/showthread.php?t=327290</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://gist.github.com/imylomylo/60e47d1902c350179a47eb3c1ffe8c5d">https://gist.github.com/imylomylo/60e47d1902c350179a47eb3c1ffe8c5d</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:paragraph -->
<p><strong><a href="https://github.com/demining/CryptoDeepTools/tree/main/37DiscreteLogarithm">Source code</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><a href="https://t.me/cryptodeeptech">Telegram: https://t.me/cryptodeeptech</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><a href="https://youtu.be/i9KYih_ffr8">Video: https://youtu.be/i9KYih_ffr8</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><a href="https://dzen.ru/video/watch/6784be61b09e46422395c236">Video tutorial: https://dzen.ru/video/watch/6784be61b09e46422395c236</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><a href="https://cryptodeeptech.ru/discrete-logarithm">Source: https://cryptodeeptech.ru/discrete-logarithm</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:heading -->
<h2 class="wp-block-heading" id="user-content-block-67e26253-470e-4432-a4e1-65b7b8b74c1b">Useful information for enthusiasts:</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p><a href="https://github.com/demining/bitcoindigger/blob/main/001_Creating_RawTX_Bitcoin_Transactions_Using_Bloom_Filter_in_Python/README.md#useful-information-for-enthusiasts"></a></p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul id="user-content-block-5e543c86-afad-430c-8aac-8ff0ffccb4e2" class="wp-block-list"><!-- wp:list-item -->
<li><strong>[1]  </strong><em><strong><a href="https://www.youtube.com/@cryptodeeptech">YouTube Channel CryptoDeepTech</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[2]  </strong><em><strong><a href="https://t.me/s/cryptodeeptech">Telegram Channel CryptoDeepTech</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[3]  </strong><a href="https://github.com/demining/CryptoDeepTools"><em><strong>GitHub Repositories  </strong></em> </a><em><strong><a href="https://github.com/demining/CryptoDeepTools">CryptoDeepTools</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[4]   </strong><em><strong><a href="https://t.me/ExploitDarlenePRO">Telegram: ExploitDarlenePRO</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[5]  </strong><em><strong><a href="https://www.youtube.com/@ExploitDarlenePRO">YouTube Channel ExploitDarlenePRO</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[6]  </strong><em><strong><a href="https://github.com/keyhunters">GitHub Repositories Keyhunters</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[7]  </strong><em><strong><a href="https://t.me/s/Bitcoin_ChatGPT">Telegram: Bitcoin ChatGPT</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[8]  </strong><strong><em><a href="https://www.youtube.com/@BitcoinChatGPT">YouTube Channel BitcoinChatGPT</a></em></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[9]  </strong><a href="https://bitcoincorewallet.ru/"><strong><em>Bitcoin Core Wallet Vulnerability</em></strong></a><a href="https://bitcoincorewallet.ru/"> </a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[10]  </strong> <strong><a href="https://btcpays.org/"><em>BTC PAYS DOCKEYHUNT</em></a></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[11]   </strong><em><strong><a href="https://dockeyhunt.com/"> DOCKEYHUNT</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[12]   </strong><em><strong><a href="https://t.me/s/DocKeyHunt">Telegram: DocKeyHunt</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[13]   </strong><em><strong><a href="https://exploitdarlenepro.com/">ExploitDarlenePRO.com</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[14]  </strong><em><strong><a href="https://github.com/demining/Dust-Attack">DUST ATTACK</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[15]  </strong><em><strong><a href="https://bitcoin-wallets.ru/">Vulnerable Bitcoin Wallets</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[16]  </strong> <em><strong><a href="https://www.youtube.com/playlist?list=PLmq8axEAGAp_kCzd9lCjX9EabJR9zH3J-">ATTACKSAFE SOFTWARE</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[17]  </strong><em><strong><a href="https://youtu.be/CzaHitewN-4"> LATTICE ATTACK</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[18]   </strong><em><strong><a href="https://github.com/demining/Kangaroo-by-JeanLucPons"> RangeNonce</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[19]   <em><a href="https://bitcoinwhoswho.ru/">BitcoinWhosWho</a></em></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[20]   <em><a href="https://coinbin.ru/">Bitcoin Wallet by Coinbin</a></em></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[21]  </strong><em><strong><a href="https://cryptodeeptech.ru/polynonce-attack/">POLYNONCE ATTACK</a></strong></em><em><strong> <a href="https://cryptodeeptech.ru/polynonce-attack/"></a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[22]  </strong> <a href="https://cold-wallets.ru/"><strong><em>Cold Wallet Vulnerability</em></strong></a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[23]  </strong> <a href="https://bitcointrezor.ru/"><strong><em>Trezor Hardware Wallet Vulnerability</em></strong></a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[24]   <a href="https://bitcoinexodus.ru/"><em>Exodus Wallet Vulnerability</em></a></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[25]  <em><a href="https://bitoncoin.org/">BITCOIN DOCKEYHUNT</a></em><em> <a href="https://bitoncoin.org/"></a></em></strong></li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:image {"linkDestination":"custom"} -->
<figure class="wp-block-image"><a href="https://cryptodeeptech.ru/discrete-logarithm/" target="_blank" rel="noreferrer noopener"><img src="https://camo.githubusercontent.com/3568257c92c143826ea99b9e75ccdf3b05ed58e05081f33d93b5aca5129d67fc/68747470733a2f2f63727970746f64656570746f6f6c2e72752f77702d636f6e74656e742f75706c6f6164732f323032342f31322f474f4c4431303331422d31303234783537362e706e67" alt="Discrete Logarithm mathematical methods and tools for recovering cryptocurrency wallets Bitcoin"/></a></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p></p>
<!-- /wp:paragraph -->
