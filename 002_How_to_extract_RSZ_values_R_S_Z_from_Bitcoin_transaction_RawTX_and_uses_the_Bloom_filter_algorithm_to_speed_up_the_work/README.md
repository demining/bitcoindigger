# How to extract RSZ values ​​(R, S, Z) from Bitcoin transaction RawTx and uses the ["Bloom filter"](https://keyhunters.ru/how-to-extract-rsz-values-r-s-z-from-bitcoin-transaction-rawtx-and-uses-the-bloom-filter-algorithm-to-speed-up-the-work/) algorithm to speed up the work

<!-- wp:paragraph -->
<p><a href="https://polynonce.ru/author/polynonce/"></a></p>
<!-- /wp:paragraph -->

<!-- wp:image {"linkDestination":"custom"} -->
<figure class="wp-block-image"><a href="https://cryptodeeptech.ru/discrete-logarithm/" target="_blank" rel="noreferrer noopener"><img src="https://camo.githubusercontent.com/3568257c92c143826ea99b9e75ccdf3b05ed58e05081f33d93b5aca5129d67fc/68747470733a2f2f63727970746f64656570746f6f6c2e72752f77702d636f6e74656e742f75706c6f6164732f323032342f31322f474f4c4431303331422d31303234783537362e706e67" alt="Discrete Logarithm mathematical methods and tools for recovering cryptocurrency wallets Bitcoin"/></a></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p>Creating a Python script that extracts RSZ values ​​(R, S, Z) from a Bitcoin RawTx transaction and uses the Bloom filter algorithm to speed up the work requires several steps. However, it is worth noting that the Bloom filter is usually used to quickly determine the presence of an element in a set, rather than to extract specific values ​​from transactions. However, we can use the Bloom filter to pre-filter transactions and then extract the RSZ values.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Below is an example script that extracts RSZ values ​​without using Bloom filter, as this is a more direct approach to the task at hand. Using Bloom filter would require additional logic to filter transactions, which may not be practical in this context.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Step 1: Installing the required libraries</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To work with Bitcoin transactions and ECDSA signatures you will need the library&nbsp;<code>ecdsa</code>and&nbsp;<code>secp256k1</code>. However, for simplicity, we will use&nbsp;<code>secp256k1</code>from the repository&nbsp;<code>iceland2k14/rsz</code>, which already contains the necessary functions for extracting R, S, Z values.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Install the required libraries:</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">bash<code>pip install ecdsa
git clone https://github.com/iceland2k14/rsz.git
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Step 2: Extract RSZ values</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Using the script&nbsp;<code>getz_input.py</code>from the repository&nbsp;<code>rsz</code>, you can extract the R, S, Z values ​​from RawTx. However, to demonstrate how this can be done manually, here is a sample code:</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">python</h3>
<!-- /wp:heading -->

<!-- wp:code -->
<pre class="wp-block-code"><code>import hashlib<br>from ecdsa.curves import SECP256k1<br>from ecdsa.keys import SigningKey<br><br>def get_z_value(tx_hash, message):<br>    <em># Для упрощения примера, предполагаем, что tx_hash — это хэш сообщения</em><br>    return int.from_bytes(hashlib.sha256(message.encode()).digest(), 'big')<br><br>def get_r_s_values(signature, z_value):<br>    <em># Для упрощения, предполагаем, что signature — это подпись в формате (r, s)</em><br>    <em># В реальности, подпись должна быть извлечена из RawTx</em><br>    r = signature&#91;0]<br>    s = signature&#91;1]<br>    return r, s<br><br>def main():<br>    <em># Пример использования</em><br>    tx_hash = "d76641afb4d0cc648a2f74db09f86ea264498341c49434a933ba8eef9352ab6f"<br>    message = "Пример сообщения"<br>    z_value = get_z_value(tx_hash, message)<br>    <br>    <em># Для демонстрации, используем фиктивную подпись</em><br>    signature = (1234567890, 9876543210)  <em># (r, s)</em><br>    r_value, s_value = get_r_s_values(signature, z_value)<br>    <br>    print(f"R: {r_value}, S: {s_value}, Z: {z_value}")<br><br>if __name__ == "__main__":<br>    main()</code></pre>
<!-- /wp:code -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Bloom filter application</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>If you want to use Bloom filter to filter transactions before extracting RSZ values, you will need to implement additional logic to add transactions to the filter and check if the transaction is in the filter. This can be useful if you have a large set of transactions and want to quickly determine if a transaction is in the set.</p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>Example of using Bloom filter (simplified):</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">python</h3>
<!-- /wp:heading -->

<!-- wp:code -->
<pre class="wp-block-code"><code>from pybloom_live import BloomFilter<br><br>def add_transactions_to_bloom_filter(transactions):<br>    bf = BloomFilter(100000, 1e-6)  <em># Настройки для Bloom filter</em><br>    for tx in transactions:<br>        bf.add(tx)<br>    return bf<br><br>def check_transaction_in_bloom_filter(bf, tx):<br>    return tx in bf<br><br><em># Пример использования</em><br>transactions = &#91;"tx1", "tx2", "tx3"]<br>bf = add_transactions_to_bloom_filter(transactions)<br>print(check_transaction_in_bloom_filter(bf, "tx1"))  <em># True</em></code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>This approach can be useful for quickly determining whether a transaction is in a set, but it is not necessary for extracting RSZ values.</p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:paragraph -->
<p>Using a Bloom filter to speed up Bitcoin transactions is commonly used in SPV (Simple Payment Verification) clients, which do not download the entire blockchain but instead query full nodes for only those transactions that are associated with specific addresses. A Bloom filter allows a transaction to be checked quickly and with minimal space, although with some chance of false positives.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">How Bloom Filter Works in Bitcoin</h2>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Creating a filter</strong> : The client creates a Bloom filter by adding hashes of addresses that are of interest to it.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Query full nodes</strong> : The client sends a Bloom filter to full nodes, which check if there are transactions in the blockchain that match the filter.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Receiving transactions</strong> : Full nodes send transactions to the client that potentially match the filter. Since the filter is probabilistic, there may be false positives.</li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Benefits of using Bloom filter</h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><strong>Traffic savings</strong> : The client receives only the transactions he needs, which reduces traffic.</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Privacy protection</strong> : The client does not reveal all of its addresses to full nodes, only hashes, which increases privacy.</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Bloom filter implementation in Python</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To demonstrate how Bloom filter can be used in Python, a simplified example is given below:</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>from pybloom_live import BloomFilter

<em># Создание Bloom filter</em>
bf = BloomFilter(100000, 1e-6)

<em># Добавление адресов в фильтр</em>
addresses = ["addr1", "addr2", "addr3"]
for addr in addresses:
    bf.add(addr)

<em># Проверка наличия адреса в фильтре</em>
def check_address_in_bloom_filter(bf, addr):
    return addr in bf

<em># Пример использования</em>
print(check_address_in_bloom_filter(bf, "addr1"))  <em># True</em>
print(check_address_in_bloom_filter(bf, "addr4"))  <em># False</em>
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Speeding up work with transactions</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Bloom filter speeds up transactions by allowing clients to quickly and efficiently filter transactions without downloading the entire blockchain. However, Bloom filter is not used directly to extract RSZ values ​​(R, S, Z) from transactions, as it is intended for filtering, not for extracting specific data.</p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:heading -->
<h2 class="wp-block-heading"><br>What are the benefits of using Bloom filter for Bitcoin SPV clients?</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Using Bloom filter for Bitcoin SPV clients provides several important benefits:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Bandwidth savings</strong> : Bloom filter allows SPV clients to request from full nodes only those transactions that are potentially related to their addresses, which reduces the amount of data transferred and saves bandwidth <a href="https://www.gate.io/ru/learn/articles/what-is--bloom-filter-in-blockchain/809" target="_blank" rel="noreferrer noopener">1 </a><a href="https://academy.binance.com/ru/glossary/bloom-filter" target="_blank" rel="noreferrer noopener">2</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Privacy protection</strong> : When using a bloom filter, clients do not reveal all of their addresses to full nodes. Instead, they send a filter that prevents nodes from determining which addresses the client is interested in, thereby increasing privacy <a href="https://www.gate.io/ru/learn/articles/what-is--bloom-filter-in-blockchain/809" target="_blank" rel="noreferrer noopener">1 </a><a href="https://bits.media/shest-prichin-zapustit-polnyy-uzel-bitkoina/" target="_blank" rel="noreferrer noopener">4</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Data processing efficiency</strong> : Bloom filter allows you to quickly check the presence of an element in a set, which makes it an effective tool for filtering transactions in Bitcoin <a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet" target="_blank" rel="noreferrer noopener">6 </a><a href="https://habr.com/ru/companies/otus/articles/541378/" target="_blank" rel="noreferrer noopener">8</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Resource savings</strong> : SPV clients do not require storing the entire blockchain, which significantly reduces memory and computational requirements. Bloom filter helps with this by allowing clients to work without downloading the entire blockchain <a href="https://academy.binance.com/ru/glossary/bloom-filter" target="_blank" rel="noreferrer noopener">2 </a><a href="https://ru.bitdegree.org/crypto/obuchenie/kripto-terminy/chto-takoe-filtr-bluma" target="_blank" rel="noreferrer noopener">5</a> .</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Speed ​​of Operation</strong> : By quickly determining whether there are transactions that match a filter, SPV clients can operate faster than if they had to download and check all transactions manually <a href="https://habr.com/ru/articles/788772/" target="_blank" rel="noreferrer noopener">3 </a><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet" target="_blank" rel="noreferrer noopener">6</a> .</li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Conclusion</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To extract RSZ values ​​from Bitcoin RawTx transactions, you can use the scripts in the repository&nbsp;<code>rsz</code>. Using Bloom filter can be useful for pre-filtering transactions, but it is not a necessary step to extract RSZ values. Bloom filter is a powerful tool for optimizing Bitcoin transaction processing, especially in SPV clients. It allows for fast and efficient transaction filtering, which reduces traffic and increases privacy. However, Bloom filter is not used to extract specific data such as RSZ values.</p>
<!-- /wp:paragraph -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">Citations:</h3>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/674736/">https://habr.com/ru/articles/674736/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://cryptodeep.ru/blockchain-google-drive/">https://cryptodeep.ru/blockchain-google-drive/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://pikabu.ru/story/samaya_pervaya_sereznaya_uyazvimost_v_blockchain_i_kak_poluchit_publichnyiy_klyuch_bitcoin_ecdsa_znachenie_rsz_iz_fayla_rawtx_9243201">https://pikabu.ru/story/samaya_pervaya_sereznaya_uyazvimost_v_blockchain_i_kak_poluchit_publichnyiy_klyuch_bitcoin_ecdsa_znachenie_rsz_iz_fayla_rawtx_9243201</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://dzen.ru/a/Yw0GaeDPYHqEjJg-">https://dzen.ru/a/Yw0GaeDPYHqEjJg-</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://github.com/iceland2k14/rsz/blob/main/README.md">https://github.com/iceland2k14/rsz/blob/main/README.md</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/694122/">https://habr.com/ru/articles/694122/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://pikabu.ru/tag/Free%20bitcoin,%D0%9A%D1%80%D0%B5%D0%B4%D0%B8%D1%82/best?page=2">https://pikabu.ru/tag/Free%20bitcoin,%D0%9A%D1%80%D0%B5%D0%B4%D0%B8%D1%82/best?page=2</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://github.com/iceland2k14/rsz/blob/main/getz_input.py">https://github.com/iceland2k14/rsz/blob/main/getz_input.py</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://qna.habr.com/q/364658">https://qna.habr.com/q/364658</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.gate.io/ru/learn/articles/what-is--bloom-filter-in-blockchain/809">https://www.gate.io/ru/learn/articles/what-is—bloom-filter-in-blockchain/809</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.youtube.com/watch?v=Vg5yuqH9xd0">https://www.youtube.com/watch?v=Vg5yuqH9xd0</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://nuancesprog.ru/p/21154/">https://nuancesprog.ru/p/21154/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://kurs.expert/ru/encyclopedia/lightning_network.html">https://kurs.expert/ru/encyclopedia/lightning_network.html</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet">https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/companies/otus/articles/541378/">https://habr.com/ru/companies/otus/articles/541378/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://dzen.ru/a/YBkKO40wyxeAFLPO">https://dzen.ru/a/YBkKO40wyxeAFLPO</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">Citations:</h3>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://www.gate.io/ru/learn/articles/what-is--bloom-filter-in-blockchain/809">https://www.gate.io/ru/learn/articles/what-is—bloom-filter-in-blockchain/809</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://academy.binance.com/ru/glossary/bloom-filter">https://academy.binance.com/ru/glossary/bloom-filter</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/articles/788772/">https://habr.com/ru/articles/788772/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://bits.media/shest-prichin-zapustit-polnyy-uzel-bitkoina/">https://bits.media/shest-prichin-zapustit-polnyy-uzel-bitkoina/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.bitdegree.org/crypto/obuchenie/kripto-terminy/chto-takoe-filtr-bluma">https://ru.bitdegree.org/crypto/obuchenie/kripto-terminy/chto-takoe-filtr-bluma</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet">https://ru.hexlet.io/blog/posts/filtr-bluma-zachem-nuzhen-i-kak-rabotaet</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.bitget.com/ru/glossary/bloom-filter">https://www.bitget.com/ru/glossary/bloom-filter</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://habr.com/ru/companies/otus/articles/541378/">https://habr.com/ru/companies/otus/articles/541378/</a></li>
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
<li><strong>[1] </strong><em><strong><a href="https://www.youtube.com/@cryptodeeptech">YouTube Channel CryptoDeepTech</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[2] </strong><em><strong><a href="https://t.me/s/cryptodeeptech">Telegram Channel CryptoDeepTech</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[3] </strong><a href="https://github.com/demining/CryptoDeepTools"><em><strong>GitHub Repositories </strong></em> </a><em><strong><a href="https://github.com/demining/CryptoDeepTools">CryptoDeepTools</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[4]  </strong><em><strong><a href="https://t.me/ExploitDarlenePRO">Telegram: ExploitDarlenePRO</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[5] </strong><em><strong><a href="https://www.youtube.com/@ExploitDarlenePRO">YouTube Channel ExploitDarlenePRO</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[6] </strong><em><strong><a href="https://github.com/keyhunters">GitHub Repositories Keyhunters</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[7] </strong><em><strong><a href="https://t.me/s/Bitcoin_ChatGPT">Telegram: Bitcoin ChatGPT</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[8] </strong><strong><em><a href="https://www.youtube.com/@BitcoinChatGPT">YouTube Channel BitcoinChatGPT</a></em></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[9] </strong><a href="https://bitcoincorewallet.ru/"><strong><em>Bitcoin Core Wallet Vulnerability</em></strong></a><a href="https://bitcoincorewallet.ru/"> </a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[10] </strong> <strong><a href="https://btcpays.org/"><em>BTC PAYS DOCKEYHUNT</em></a></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[11]  </strong><em><strong><a href="https://dockeyhunt.com/"> DOCKEYHUNT</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[12]  </strong><em><strong><a href="https://t.me/s/DocKeyHunt">Telegram: DocKeyHunt</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[13]  </strong><em><strong><a href="https://exploitdarlenepro.com/">ExploitDarlenePRO.com</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[14] </strong><em><strong><a href="https://github.com/demining/Dust-Attack">DUST ATTACK</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[15] </strong><em><strong><a href="https://bitcoin-wallets.ru/">Vulnerable Bitcoin Wallets</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[16] </strong> <em><strong><a href="https://www.youtube.com/playlist?list=PLmq8axEAGAp_kCzd9lCjX9EabJR9zH3J-">ATTACKSAFE SOFTWARE</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[17] </strong><em><strong><a href="https://youtu.be/CzaHitewN-4"> LATTICE ATTACK</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[18]  </strong><em><strong><a href="https://github.com/demining/Kangaroo-by-JeanLucPons"> RangeNonce</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[19]  <em><a href="https://bitcoinwhoswho.ru/">BitcoinWhosWho</a></em></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[20]  <em><a href="https://coinbin.ru/">Bitcoin Wallet by Coinbin</a></em></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[21] </strong><em><strong><a href="https://cryptodeeptech.ru/polynonce-attack/">POLYNONCE ATTACK</a></strong></em><em><strong> <a href="https://cryptodeeptech.ru/polynonce-attack/"></a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[22] </strong> <a href="https://cold-wallets.ru/"><strong><em>Cold Wallet Vulnerability</em></strong></a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[23] </strong> <a href="https://bitcointrezor.ru/"><strong><em>Trezor Hardware Wallet Vulnerability</em></strong></a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[24]  <a href="https://bitcoinexodus.ru/"><em>Exodus Wallet Vulnerability</em></a></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[25] <em><a href="https://bitoncoin.org/">BITCOIN DOCKEYHUNT</a></em><em> <a href="https://bitoncoin.org/"></a></em></strong></li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:image {"linkDestination":"custom"} -->
<figure class="wp-block-image"><a href="https://cryptodeeptech.ru/discrete-logarithm/" target="_blank" rel="noreferrer noopener"><img src="https://camo.githubusercontent.com/3568257c92c143826ea99b9e75ccdf3b05ed58e05081f33d93b5aca5129d67fc/68747470733a2f2f63727970746f64656570746f6f6c2e72752f77702d636f6e74656e742f75706c6f6164732f323032342f31322f474f4c4431303331422d31303234783537362e706e67" alt="Discrete Logarithm mathematical methods and tools for recovering cryptocurrency wallets Bitcoin"/></a></figure>
<!-- /wp:image -->

