# Creating RawTX Bitcoin Transactions Using Bloom Filter in Python

<!-- wp:image {"id":2509,"sizeSlug":"large","linkDestination":"none","align":"center"} -->
<figure class="wp-block-image aligncenter size-large"><img src="https://keyhunters.ru/wp-content/uploads/2025/03/visual-selection-2-1024x666.png" alt="" class="wp-image-2509"/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><a href="https://polynonce.ru/author/polynonce/"></a></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p>To create a raw Bitcoin transaction using Bloom filter in Python, we combine two key technologies: transaction formation and data filtering. Here is the implementation:</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">python</h2>
<!-- /wp:heading -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted"><code>from bitcoinutils.setup import setup<br>from bitcoinutils.transactions import Transaction, TxInput, TxOutput<br>from bitcoinutils.keys import PrivateKey, P2pkhAddress<br>from pybloom import BloomFilter<br><br>def create_raw_tx():<br>    <em># Инициализация сети Bitcoin</em><br>    setup('mainnet')<br><br>    <em># Инициализация Bloom filter с адресами кошелька</em><br>    wallet_addresses = &#91;<br>        '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',<br>        '1BitcoinEaterAddressDontSendf59kuE'<br>    ]<br>    <br>    bloom = BloomFilter(<br>        capacity=1000, <br>        error_rate=0.001, <br>        backend='bitarray'<br>    )<br>    <br>    for addr in wallet_addresses:<br>        bloom.add(addr)<br><br>    <em># Фильтрация UTXO (пример данных)</em><br>    utxo_list = &#91;<br>        {<br>            'txid': 'abc123...',<br>            'vout': 0,<br>            'amount': 0.01,<br>            'address': '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',<br>            'privkey': 'L4gB7...'<br>        }<br>    ]<br>    <br>    <em># Выбор подходящих UTXO через Bloom filter</em><br>    selected_utxo = &#91;<br>        utxo for utxo in utxo_list <br>        if utxo&#91;'address'] in bloom<br>    ]<br><br>    <em># Создание входов транзакции</em><br>    inputs = &#91;]<br>    for utxo in selected_utxo:<br>        inputs.append(TxInput(utxo&#91;'txid'], utxo&#91;'vout']))<br><br>    <em># Создание выходов</em><br>    output = TxOutput(<br>        0.009,  <em># сумма с учетом комиссии</em><br>        P2pkhAddress('recipient_address_here').to_script_pub_key()<br>    )<br><br>    <em># Сборка транзакции</em><br>    tx = Transaction(inputs, &#91;output])<br><br>    <em># Подписание входов</em><br>    for i, utxo in enumerate(selected_utxo):<br>        priv_key = PrivateKey.from_wif(utxo&#91;'privkey'])<br>        tx.sign_input(i, priv_key)<br><br>    return tx.serialize()<br><br><em># Выполнение</em><br>print("Raw transaction:", create_raw_tx())</code></pre>
<!-- /wp:preformatted -->

<!-- wp:paragraph -->
<p><strong>Key components of the solution:</strong></p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Bloom Filter initialization</strong><br>Used to filter UTXOs belonging to wallet <a href="https://bitcoinops.org/en/topics/transaction-bloom-filtering/" target="_blank" rel="noreferrer noopener">5 </a><a href="https://hackernoon.com/enhancing-bitcoins-transaction-privacy-with-bloom-filters-2m5q33ta" target="_blank" rel="noreferrer noopener">6</a> . Parameters:<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li><code>capacity</code>— expected number of elements</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><code>error_rate</code>— acceptable probability of false positives</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>UTXO filtering mechanism</strong><br>Checking address ownership via operator <code>in</code><a href="https://pypi.org/project/BloomFilterPy/" target="_blank" rel="noreferrer noopener">3 </a><a href="https://pypi.org/project/bloom-filter/" target="_blank" rel="noreferrer noopener">4</a> . Actual data should be obtained from:<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Local UTXO base</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>External Blockchain APIs</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>Bitcoin Full Node</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Optimizing the</strong><br>Bloom filter speed reduces the UTXO lookup time from O(n) to O(1) <a href="https://hackernoon.com/enhancing-bitcoins-transaction-privacy-with-bloom-filters-2m5q33ta" target="_blank" rel="noreferrer noopener">6</a> , which is critical for:<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Wallets with a large transaction history</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>Real time systems</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>Mobile applications</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list --></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p><strong>Recommendations for improvement:</strong></p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li>Add change output processing</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>Implement dynamic commission calculation</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>Integrate Bitcoin node connection via JSON-RPC</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>Add support for SegWit addresses</li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>The script requires the following dependencies to work:</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>pip install bitcoinutils BloomFilterPy</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p>Example output:</p>
<!-- /wp:paragraph -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted"><code>Raw transaction: 0200000001... [полный hex транзакции]</code><br><br>To create a raw Bitcoin transaction using BloomFilterPy, follow this step-by-step guide. The example implements optimized UTXO lookup and transaction assembly. <br><br>**1. Installing Dependencies** <br>```bash <br>pip install bitcoinutils BloomFilterPy <br>``` <br><br>**2. Full example script** <br>```python <br>from bitcoinutils.setup import setup <br>from bitcoinutils.transactions import Transaction, TxInput, TxOutput <br>from bitcoinutils.keys import PrivateKey, P2pkhAddress <br>from pybloom import BloomFilter <br><br>def build_tx_with_bloom(): <br>    # Initialize the Bitcoin network <br>    setup('mainnet') <br>    <br>    # List of wallet addresses <br>    wallet_addrs = [ <br>        '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa', <br>        'bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq' <br>    ] <br>    <br>    # Initialize Bloom filter <br>    bloom = BloomFilter( <br>        capacity=500, # Expected number of elements <br>        error_rate=0.01, # 1% probability of errors <br>        mode='optimal' # Automatic selection of parameters <br>    ) <br>    <br>    # Adding addresses to the filter <br>    for addr in wallet_addrs: <br>        bloom.add(addr.encode('utf-8')) # Be sure to convert to bytes <br><br>    # Getting UTXO (sample data) <br>    utxo_pool = [ <br>        { <br>            'txid': 'a9d459...', <br>            'vout': 0, <br>            'amount': 0.005, <br>            'address': '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa', <br>            'privkey': 'L4gB7...' <br>        } <br>    ] <br>    <br>    # Filtering via Bloom <br>    valid_utxo = [utxo for utxo in utxo_pool <br>                 if bloom.check(utxo['address'].encode('utf-8'))] <br><br>    # Creating transaction inputs <br>    tx_inputs = [TxInput(utxo['txid'], utxo['vout']) for utxo in valid_utxo] <br><br>    # Calculate amount and fee <br>    total_in = sum(utxo['amount'] for utxo in valid_utxo) <br>    fee = 0.00015 # Fee example <br>    send_amount = total_in - fee <br><br>    # Create output <br>    recipient = '1BoatSLRHtKNngkdXEeobR76b53LETtpyT' <br>    tx_output = TxOutput( <br>        send_amount, <br>        P2pkhAddress(recipient).to_script_pub_key() <br>    ) <br><br>    # Assemble and sign <br>    tx = Transaction(tx_inputs, [tx_output]) <br>    for i, utxo in enumerate(valid_utxo): <br>        tx.sign_input(i, PrivateKey.from_wif(utxo['privkey'])) <br><br>    return tx.serialize() <br><br>print(f"Raw TX:{build_tx_with_bloom()}") <br>``` <br><br>**Key aspects of working with BloomFilterPy:**<br><br>| Parameter | Recommended value | Description | <br>|-------------------|-------------------------|-----------------------------------| <br>| `capacity` | N × 1.2 | N - expected number of elements | <br>| `error_rate` | 0.01-0.001 | Filtering accuracy | <br>| `hash_seeds` | Auto-select | Seeds for hash functions | <br>| `mode` | `optimal` | Automatic optimization | <br><br>**Performance optimizations:** <br>1. Use `bloom.check()` instead of `in` operator for checking <br>2. Convert strings to bytes before adding to filter <br>3. For large datasets, increase `capacity` by 20-30% <br><br>**Security recommendations:** <br>```python <br># Additional check after filtering <br>from bitcoinutils.utils import is_valid_address <br><br>verified_utxo = [] <br>for utxo in valid_utxo: <br>    if is_valid_address(utxo['address']): <br>        verified_utxo.append(utxo) <br>``` <br><br>**Typical output:** <br>``` <br>Raw TX: 0100000001... [full HEX of transaction] <br>``` <br><br>For real use case you need to: <br>1. Integrate UTXO source (Electrum server, Blockstream API) <br>2. Implement dynamic fee calculation via mempool.space API <br>3. Add change address processing <br>4. Enable SegWit support (bech32 addresses) <br><br>Example of calculating Bloom filter parameters for 1000 addresses: <br>```python <br>BloomFilter( <br>    capacity=1200, <br>    error_rate=0.001, <br>    mode='fast' # Speed ​​optimization <br>) <br>```<br><br><br><br><br><br><br><code></code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">How to Integrate BloomFilterPy with Other Libraries to Create RawTX</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>To integrate BloomFilterPy with various Bitcoin transaction creation libraries, we will look at three popular frameworks. Each example demonstrates the key steps: UTXO filtering and transaction assembly.</p>
<!-- /wp:paragraph -->

<!-- wp:heading -->
<h2 class="wp-block-heading">1. Integration with&nbsp;<strong>bitcoinlib</strong>&nbsp;(HD wallets)</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>python</p>
<!-- /wp:paragraph -->

<!-- wp:code -->
<pre class="wp-block-code"><code>from bitcoinlib.transactions import Transaction<br>from bitcoinlib.keys import Key<br>from pybloom import BloomFilter<br><br><em># Инициализация Bloom фильтра</em><br>bloom = BloomFilter(capacity=1000, error_rate=0.01)<br>wallet_addresses = &#91;'1A1zP...', 'bc1q...']<br>&#91;bloom.add(addr.encode()) for addr in wallet_addresses]<br><br><em># Фильтрация UTXO через Electrum-сервер</em><br>from bitcoinlib.services.services import Service<br><br>service = Service(network='bitcoin')<br>utxos = service.getutxos(wallet_addresses&#91;0])<br><br>filtered_utxos = &#91;utxo for utxo in utxos <br>                 if bloom.check(utxo&#91;'address'].encode())]<br><br><em># Создание транзакции</em><br>tx = Transaction(network='bitcoin')<br>for utxo in filtered_utxos:<br>    tx.add_input(utxo&#91;'txid'], utxo&#91;'output_n'])<br><br>tx.add_output(0.01, 'recipient_address')<br><br><em># Подписание</em><br>key = Key.from_passphrase('your_wallet_passphrase')<br>tx.sign(key)<br><br>print("Raw TX:", tx.raw_hex())</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p><strong>Peculiarities:</strong></p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Built-in work with Electrum servers</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>BIP32 HD wallet support</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>Automatic calculation of commissions via<code>.fee_per_kb</code></li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">2. Integration with&nbsp;<strong>pycoin</strong>&nbsp;(advanced functionality)</h2>
<!-- /wp:heading -->

<!-- wp:code -->
<pre class="wp-block-code"><code>from pycoin.tx.Tx import Tx<br>from pycoin.tx.tx_utils import create_tx<br>from pycoin.ui import standard_tx_out_script<br>from pybloom import BloomFilter<br><br><em># Конфигурация Bloom</em><br>bloom = BloomFilter(capacity=500, error_rate=0.001)<br>known_hashes = {b'addr1_hash', b'addr2_hash'}<br>&#91;bloom.add(h) for h in known_hashes]<br><br><em># Получение UTXO через собственный узел</em><br>def get_filtered_utxos(node):<br>    return &#91;utxo for utxo in node.getutxos() <br>           if bloom.check(hash(utxo.address))]<br><br><em># Сборка транзакции</em><br>tx = create_tx(<br>    inputs=get_filtered_utxos(my_node),<br>    outputs=&#91;(recipient_addr, 0.01)],<br>    fee=0.0001,<br>    network='BTC'<br>)<br><br>print("Raw TX:", tx.as_hex())</code></pre>
<!-- /wp:code -->

<!-- wp:paragraph -->
<p><strong>Optimizations:</strong></p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Using address hashes instead of strings</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>Batch check via<code>bloom.batch_check()</code></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>PSBT support for multi-signatures</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:heading -->
<h2 class="wp-block-heading">3. Integration with&nbsp;<strong>bit</strong>&nbsp;(simple library)</h2>
<!-- /wp:heading -->

<!-- wp:preformatted -->
<pre class="wp-block-preformatted">python<code>from bit import Key, NetworkAPI
from bit.transaction import prepare_transaction
from pybloom import BloomFilter

<em># Инициализация фильтра</em>
bloom = BloomFilter(100, 0.01)
key = Key()
bloom.add(key.address.encode())

<em># Фильтрация UTXO</em>
utxos = NetworkAPI.get_unspent(key.address)
inputs = [utxo for utxo in utxos 
         if bloom.check(utxo['address'].encode())]

<em># Сборка транзакции</em>
outputs = [('1BoatSLRHtKNngkdXEeobR76b53LETtpyT', 0.01, 'btc')]
raw_tx = prepare_transaction(
    inputs=inputs,
    outputs=outputs,
    leftover=key.address  <em># Автоматический change-адрес</em>
)

print("Raw TX:", raw_tx)
</code></pre>
<!-- /wp:preformatted -->

<!-- wp:heading -->
<h2 class="wp-block-heading">Comparison of approaches</h2>
<!-- /wp:heading -->

<!-- wp:table -->
<figure class="wp-block-table"><table class="has-fixed-layout"><thead><tr><th>Library</th><th>Advantages</th><th>Restrictions</th></tr></thead><tbody><tr><td><strong>bitcoinlib</strong></td><td>BIP39/44 support, multi-network</td><td>Complex setup</td></tr><tr><td><strong>pycoin</strong></td><td>Flexible signature scripts</td><td>Manual assembly required</td></tr><tr><td><strong>bit</strong></td><td>Simple API, automatic commissions</td><td>Basic scenarios only</td></tr></tbody></table></figure>
<!-- /wp:table -->

<!-- wp:heading -->
<h2 class="wp-block-heading">General recommendations</h2>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><strong>Optimizing Bloom parameters</strong> python<code>bloom = BloomFilter( capacity=len(addresses) * 1.3, <em># 30% запас</em> error_rate=0.001, mode='fast' )</code></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Safe UTXO handling</strong> python<code>def verify_utxo(utxo): return ( bloom.check(utxo.address) and validate_signature(utxo) and check_double_spend(utxo) )</code></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>Integration with Bitcoin Core</strong> python node<code>from bitcoin.rpc import RawProxy proxy = RawProxy() raw_utxos = proxy.listunspent(0)</code></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>python <strong>batch processing</strong><code>addresses_batch = [addr.encode() for addr in 1000_addresses] bloom.batch_add(addresses_batch)</code></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>For industrial use it is recommended:</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul class="wp-block-list"><!-- wp:list-item -->
<li>Use C extensions like<code>pybloomfiltermmap3</code></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>Implement filter caching</li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li>Add false positive monitoring</li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:heading {"level":3} -->
<h3 class="wp-block-heading">Citations:</h3>
<!-- /wp:heading -->

<!-- wp:list {"ordered":true} -->
<ol class="wp-block-list"><!-- wp:list-item -->
<li><a href="https://github.com/Bit-Wasp/bitcoin-php/issues/238">https://github.com/Bit-Wasp/bitcoin-php/issues/238</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://pypi.org/project/python-bitcoin-tools/">https://pypi.org/project/python-bitcoin-tools/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://pypi.org/project/BloomFilterPy/">https://pypi.org/project/BloomFilterPy/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://pypi.org/project/bloom-filter/">https://pypi.org/project/bloom-filter/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://bitcoinops.org/en/topics/transaction-bloom-filtering/">https://bitcoinops.org/en/topics/transaction-bloom-filtering/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://hackernoon.com/enhancing-bitcoins-transaction-privacy-with-bloom-filters-2m5q33ta">https://hackernoon.com/enhancing-bitcoins-transaction-privacy-with-bloom-filters-2m5q33ta</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://gist.github.com/haidoan/26d76320e411887cdada18fec86bb333">https://gist.github.com/haidoan/26d76320e411887cdada18fec86bb333</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://bitcoin.stackexchange.com/questions/92957/how-to-parse-transaction-script-to-address-the-correct-way">https://bitcoin.stackexchange.com/questions/92957/how-to-parse-transaction-script-to-address-the-correct-way</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://pypi.org/project/btctxstore/0.1.0/">https://pypi.org/project/btctxstore/0.1.0/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://bitcoin.stackexchange.com/questions/38480/are-there-any-python-modules-that-decode-raw-transaction-data">https://bitcoin.stackexchange.com/questions/38480/are-there-any-python-modules-that-decode-raw-transaction-data</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://bitcoin.stackexchange.com/questions/115263/bitcoin-rpc-createrawtransaction-parse-error">https://bitcoin.stackexchange.com/questions/115263/bitcoin-rpc-createrawtransaction-parse-error</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://bitcoin.stackexchange.com/questions/30550/is-it-possible-to-create-a-transaction-in-pure-python-without-needing-to-run-bit">https://bitcoin.stackexchange.com/questions/30550/is-it-possible-to-create-a-transaction-in-pure-python-without-needing-to-run-bit</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.youtube.com/watch?v=qZNJTh2NEiU">https://www.youtube.com/watch?v=qZNJTh2NEiU</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://python.land/bloom-filter">https://python.land/bloom-filter</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://pypi.org/project/shaped-bloom-filter/">https://pypi.org/project/shaped-bloom-filter/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://gist.github.com/marcan/23e1ec416bf884dcd7f0e635ce5f2724">https://gist.github.com/marcan/23e1ec416bf884dcd7f0e635ce5f2724</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://llimllib.github.io/bloomfilter-tutorial/">https://llimllib.github.io/bloomfilter-tutorial/</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://bitcoin.stackexchange.com/questions/37124/is-there-a-way-to-index-transactions-so-that-filterload-commands-can-be-answered">https://bitcoin.stackexchange.com/questions/37124/is-there-a-way-to-index-transactions-so-that-filterload-commands-can-be-answered</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://ethz.ch/content/dam/ethz/special-interest/infk/inst-infsec/system-security-group-dam/research/publications/pub2014/acsac_gervais.pdf">https://ethz.ch/content/dam/ethz/special-interest/infk/inst-infsec/system-security-group-dam/research/publications/pub2014/acsac_gervais.pdf</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://developer.bitcoin.org/examples/p2p_networking.html?highlight=bloom+filter">https://developer.bitcoin.org/examples/p2p_networking.html?highlight=bloom+filter</a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><a href="https://www.npmjs.com/package/@synonymdev/raw-transaction-decoder">https://www.npmjs.com/package/@synonymdev/raw-transaction-decoder</a></li>
<!-- /wp:list-item --></ol>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:paragraph -->
<p><strong><a href="https://github.com/demining/CryptoDeepTools/tree/main/37DiscreteLogarithm" target="_blank" rel="noreferrer noopener">Source code</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><a href="https://t.me/cryptodeeptech" target="_blank" rel="noreferrer noopener">Telegram: https://t.me/cryptodeeptech</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><a href="https://youtu.be/i9KYih_ffr8" target="_blank" rel="noreferrer noopener">Video: https://youtu.be/i9KYih_ffr8</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><a href="https://dzen.ru/video/watch/6784be61b09e46422395c236" target="_blank" rel="noreferrer noopener">Video tutorial: https://dzen.ru/video/watch/6784be61b09e46422395c236</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:paragraph -->
<p><strong><a href="https://cryptodeeptech.ru/discrete-logarithm" target="_blank" rel="noreferrer noopener">Source: https://cryptodeeptech.ru/discrete-logarithm</a></strong></p>
<!-- /wp:paragraph -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:heading -->
<h2 class="wp-block-heading" id="block-67e26253-470e-4432-a4e1-65b7b8b74c1b">Useful information for enthusiasts:</h2>
<!-- /wp:heading -->

<!-- wp:list -->
<ul id="block-5e543c86-afad-430c-8aac-8ff0ffccb4e2" class="wp-block-list"><!-- wp:list-item -->
<li><strong>[1]</strong><em><strong><a href="https://www.youtube.com/@cryptodeeptech" target="_blank" rel="noreferrer noopener">YouTube Channel CryptoDeepTech</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[2]</strong><em><strong><a href="https://t.me/s/cryptodeeptech" target="_blank" rel="noreferrer noopener">Telegram Channel CryptoDeepTech</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[3]</strong><a href="https://github.com/demining/CryptoDeepTools" target="_blank" rel="noreferrer noopener"><em><strong>GitHub Repositories</strong></em> </a><em><strong><a href="https://github.com/demining/CryptoDeepTools" target="_blank" rel="noreferrer noopener">CryptoDeepTools</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[4] </strong><em><strong><a href="https://t.me/ExploitDarlenePRO" target="_blank" rel="noreferrer noopener">Telegram: ExploitDarlenePRO</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[5]</strong><em><strong><a href="https://www.youtube.com/@ExploitDarlenePRO" target="_blank" rel="noreferrer noopener">YouTube Channel ExploitDarlenePRO</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[6]</strong><em><strong><a href="https://github.com/keyhunters" target="_blank" rel="noreferrer noopener">GitHub Repositories Keyhunters</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[7]</strong><em><strong><a href="https://t.me/s/Bitcoin_ChatGPT" target="_blank" rel="noreferrer noopener">Telegram: Bitcoin ChatGPT</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[8]</strong><strong><em><a href="https://www.youtube.com/@BitcoinChatGPT" target="_blank" rel="noreferrer noopener">YouTube Channel BitcoinChatGPT</a></em></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[9]</strong><a href="https://bitcoincorewallet.ru/" target="_blank" rel="noreferrer noopener"> <strong><em>Bitcoin Core Wallet Vulnerability</em></strong></a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[10]</strong> <strong><a href="https://btcpays.org/" target="_blank" rel="noreferrer noopener"><em>BTC PAYS DOCKEYHUNT</em></a></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[11] </strong><em><strong><a href="https://dockeyhunt.com/" target="_blank" rel="noreferrer noopener"> DOCKEYHUNT</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[12] </strong><em><strong><a href="https://t.me/s/DocKeyHunt" target="_blank" rel="noreferrer noopener">Telegram: DocKeyHunt</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[13] </strong><em><strong><a href="https://exploitdarlenepro.com/" target="_blank" rel="noreferrer noopener">ExploitDarlenePRO.com</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[14]</strong><em><strong><a href="https://github.com/demining/Dust-Attack" target="_blank" rel="noreferrer noopener">DUST ATTACK</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[15]</strong><em><strong><a href="https://bitcoin-wallets.ru/" target="_blank" rel="noreferrer noopener">Vulnerable Bitcoin Wallets</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[16]</strong> <em><strong><a href="https://www.youtube.com/playlist?list=PLmq8axEAGAp_kCzd9lCjX9EabJR9zH3J-" target="_blank" rel="noreferrer noopener">ATTACKSAFE SOFTWARE</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[17]</strong><em><strong><a href="https://youtu.be/CzaHitewN-4" target="_blank" rel="noreferrer noopener"> LATTICE ATTACK</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[18] </strong><em><strong><a href="https://github.com/demining/Kangaroo-by-JeanLucPons" target="_blank" rel="noreferrer noopener"> RangeNonce</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[19] <em><a href="https://bitcoinwhoswho.ru/" target="_blank" rel="noreferrer noopener">BitcoinWhosWho</a></em></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[20] <em><a href="https://coinbin.ru/" target="_blank" rel="noreferrer noopener">Bitcoin Wallet by Coinbin</a></em></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[21]</strong><em><strong> <a href="https://cryptodeeptech.ru/polynonce-attack/" target="_blank" rel="noreferrer noopener">POLYNONCE ATTACK</a></strong></em></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[22]</strong> <a href="https://cold-wallets.ru/" target="_blank" rel="noreferrer noopener"><strong><em>Cold Wallet Vulnerability</em></strong></a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[23]</strong> <a href="https://bitcointrezor.ru/" target="_blank" rel="noreferrer noopener"><strong><em>Trezor Hardware Wallet Vulnerability</em></strong></a></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[24] <a href="https://bitcoinexodus.ru/" target="_blank" rel="noreferrer noopener"><em>Exodus Wallet Vulnerability</em></a></strong></li>
<!-- /wp:list-item -->

<!-- wp:list-item -->
<li><strong>[25]<em> <a href="https://bitoncoin.org/" target="_blank" rel="noreferrer noopener">BITCOIN DOCKEYHUNT</a></em></strong></li>
<!-- /wp:list-item --></ul>
<!-- /wp:list -->

<!-- wp:separator -->
<hr class="wp-block-separator has-alpha-channel-opacity"/>
<!-- /wp:separator -->

<!-- wp:image {"id":5107} -->
<figure class="wp-block-image"><img src="https://cryptodeeptool.ru/wp-content/uploads/2024/12/GOLD1031B-1024x576.png" alt="Discrete Logarithm mathematical methods and tools for recovering cryptocurrency wallets Bitcoin" class="wp-image-5107"/></figure>
<!-- /wp:image -->

<!-- wp:paragraph -->
<p><a href="https://www.facebook.com/sharer.php?u=https%3A%2F%2Fpolynonce.ru%2F%25d1%2581%25d0%25be%25d0%25b7%25d0%25b4%25d0%25b0%25d0%25bd%25d0%25b8%25d1%258f-rawtx-%25d1%2582%25d1%2580%25d0%25b0%25d0%25bd%25d0%25b7%25d0%25b0%25d0%25ba%25d1%2586%25d0%25b8%25d0%25b8-bitcoin-%25d1%2581-%25d0%25b8%25d1%2581%25d0%25bf%25d0%25be%25d0%25bb%25d1%258c%25d0%25b7%25d0%25be%25d0%25b2%25d0%25b0%25d0%25bd%2F" target="_blank" rel="noreferrer noopener"></a></p>
<!-- /wp:paragraph -->
