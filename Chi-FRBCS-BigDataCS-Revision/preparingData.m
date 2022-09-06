% Sampling / enumarating categorical features / spliting data / creating
% .txt files from data

clc
i = 1;  
w = 0;
file = '..\Implementation\Datasets\KDD\kddcup.data_10_percent_corrected';
fid = fopen(file);
kddcup_10_normal_versus_R2L = fopen('./kddcup_10_normal_versus_R2L.txt', 'w');
kddcup_10_normal_versus_R2L_trainFileId = fopen('./kddcup_10_normal_versus_R2L_trainData.txt', 'w');
kddcup_10_normal_versus_R2L_testFileId = fopen('./kddcup_10_normal_versus_R2L_testData.txt', 'w');
N1 = 0;
N2 = 0;
N3 = 0;
N4 = 0;
N5 = 0;
N6 = 0;
N7 = 0;
N8 = 0;
N9 = 0;
next = fgetl(fid);
% while(next ~= -1)
%     if contains(next, 'normal.', 'IgnoreCase', true) 
%         N1 = N1 + 1;
%     elseif contains(next, 'ftp_write.', 'IgnoreCase', true)
%         N2 = N2 + 1;
%     elseif contains(next, 'guess_passwd.', 'IgnoreCase', true)
%         N3 = N3 + 1;
%     elseif contains(next, 'imap.', 'IgnoreCase', true)
%         N4 = N4 + 1;
%     elseif contains(next, 'multihop.', 'IgnoreCase', true)
%         N5 = N5 + 1;
%     elseif contains(next, 'phf.', 'IgnoreCase', true) 
%         N6 = N6 + 1;
%     elseif contains(next, 'spy.', 'IgnoreCase', true)
%         N7 = N7 +1;
%     elseif contains(next, 'warezclient.', 'IgnoreCase', true)
%         N8 = N8 + 1;
%     elseif contains(next, 'warezmaster.', 'IgnoreCase', true)
%         N9 = N9 + 1;
%     end
%     next = fgetl(fid);
% end
while (next ~= -1)
    if contains(next, 'normal.', 'IgnoreCase', true) ||...
contains(next, 'warezclient.', 'IgnoreCase', true)
%         contains(next, 'ftp_write.', 'IgnoreCase', true) ||...
%         contains(next, 'guess_passwd.', 'IgnoreCase', true) ||...
%         contains(next, 'imap.', 'IgnoreCase', true) ||...
%         contains(next, 'multihop.', 'IgnoreCase', true) ||...
%         contains(next, 'phf.', 'IgnoreCase', true) ||...
%         contains(next, 'spy.', 'IgnoreCase', true) ||...
        
%     ||...
%         contains(next, 'warezmaster.', 'IgnoreCase', true)
            %% ################################################
            next = replace(next, 'tcp,', '1,');
            next = replace(next, 'udp,', '2,');
            next = replace(next, 'icmp,', '3,');
            
            %% ################################################
            next = replace(next, 'ecr_i,', '1,');
            next = replace(next, 'private,', '2,');
            next = replace(next, 'http,', '3,');
            next = replace(next, 'other,', '4,');
            next = replace(next, 'smtp,', '4,');
            next = replace(next, 'X11,', '5,');
            next = replace(next, 'urh_i,', '5,');
            next = replace(next, 'netbios_dgm,', '5,');
            next = replace(next, 'pop_2,', '5,');
            next = replace(next, 'nnsp,', '5,');
            next = replace(next, 'supdup,', '5,');
            next = replace(next, 'nntp,', '5,');
            next = replace(next, 'courier,', '5,');
            next = replace(next, 'domain,', '5,');
            next = replace(next, 'discard,', '5,');
            next = replace(next, 'telnet,', '5,');
            next = replace(next, 'urp_i,', '5,');
            next = replace(next, 'tim_i,', '5,');
            next = replace(next, 'IRC,', '5,');
            next = replace(next, 'exec,', '5,');
            next = replace(next, 'ldap,', '5,');
            next = replace(next, 'ssh,', '5,');
            next = replace(next, 'uucp,', '5,');
            next = replace(next, 'netbios_ssn,', '5,');
            next = replace(next, 'printer,', '5,');
            next = replace(next, 'iso_tsap,', '5,');
            next = replace(next, 'gopher,', '5,');
            next = replace(next, 'ntp_u,', '5,');
            next = replace(next, 'finger,', '5,');
            next = replace(next, 'red_i,', '5,');
            next = replace(next, 'Z39_50,', '5,');
            next = replace(next, 'http_443,', '5,');
            next = replace(next, 'link,', '5,');
            next = replace(next, 'hostnames,', '5,');
            next = replace(next, 'klogin,', '5,');
            next = replace(next, 'sunrpc,', '5,');
            next = replace(next, 'whois,', '5,');
            next = replace(next, 'systat,', '5,');
            next = replace(next, 'imap4,', '5,');
            next = replace(next, 'auth,', '5,');
            next = replace(next, 'ftp,', '5,');
            next = replace(next, 'tftp_u,', '5,');
            next = replace(next, 'netstat,', '5,');
            next = replace(next, 'kshell,', '5,');
            next = replace(next, 'netbios_ns,', '5,');
            next = replace(next, 'login,', '5,');
            next = replace(next, 'uucp_path,', '5,');
            next = replace(next, 'mtp,', '5,');
            next = replace(next, 'sql_net,', '5,');
            next = replace(next, 'echo,', '5,');
            next = replace(next, 'remote_job,', '5,');
            next = replace(next, 'pop_3,', '5,');
            next = replace(next, 'eco_i,', '5,');
            next = replace(next, 'pm_dump,', '5,');
            next = replace(next, 'ctf,', '5,');
            next = replace(next, 'name,', '5,');
            next = replace(next, 'daytime,', '5,');
            next = replace(next, 'efs,', '5,');
            next = replace(next, 'vmnet,', '5,');
            next = replace(next, 'bgp,', '5,');
            next = replace(next, 'rje,', '5,');
            next = replace(next, 'shell,', '5,');
            next = replace(next, 'csnet_ns,', '5,');
            next = replace(next, 'time,', '5,');
            next = replace(next, 'ftp_data,', '5,');
            next = replace(next, 'domain_u,', '5,');

            %% ################################################
            next = replace(next, 'S0,', '1,');
            next = replace(next, 'RSTR,', '2,');
            next = replace(next, 'SF,', '3,');
            next = replace(next, 'REJ,', '4,');
            next = replace(next, 'S1,', '5,');
            next = replace(next, 'S2,', '5,');
            next = replace(next, 'S3,', '5,');
            next = replace(next, 'RSTO,', '5,');
            next = replace(next, 'RSTOS0,', '5,');
            next = replace(next, 'OTH,', '5,');
            next = replace(next, 'SH,', '5,');
            
            %% ################################################
            next = replace(next, 'normal.', '1.');
            next = replace(next, 'ftp_write.', '2.');
            next = replace(next, 'guess_passwd.', '2.');
            next = replace(next, 'imap.', '2.');
            next = replace(next, 'multihop.', '2.');
            next = replace(next, 'phf.', '2.');
            next = replace(next, 'spy.', '2.');
            next = replace(next, 'warezclient.', '2.');
            next = replace(next, 'warezmaster.', '2.');
            
            %% ################################################
            
            fprintf(kddcup_10_normal_versus_R2L, '%s\n', next);
            
            if rem(w, 3) == 0
                fprintf(kddcup_10_normal_versus_R2L_testFileId, '%s\n', next);
            else
                fprintf(kddcup_10_normal_versus_R2L_trainFileId, '%s\n', next);
            end  
            w = w + 1;
    end
    i = i + 1;
    next = fgetl(fid);
end

fclose('all');
