function FeatureMap = FeatureMap(x,k)

                    length=size(x);
                    phi=ones(length(1),1);
                    for j=1:(k-1)
                        phi=[phi x.^j];
                    end
                     FeatureMap = phi;

end